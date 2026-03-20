import spinal.core.sim._
import spinal.core._
import spinal.lib.bus.amba4.axi.sim.{AxiMemorySim, AxiMemorySimConfig}
import spinal.lib.bus.amba4.axi._
import spinal.lib.bus.amba4.axilite.{AxiLite4, AxiLite4SpecRenamer}
import spinal.lib._
import utils._

import scala.language.postfixOps

object CustomBlackboxAxiRenamer {
  def apply(that: BlackboxAxi, prefix: String): Unit = {
    def doIt(): Unit = {
      that.flatten.foreach(bt => {
        val name = bt.getName()
        var newName = name
        
        // Remove "blackbox_axi_" prefix if present
        newName = newName.replace("blackbox_axi_", "")
        
        // Remove "m_axi_gmem_" prefix - SpinalHDL will add the bundle name (prefix) automatically
        newName = newName.replace("m_axi_gmem_", "")
        
        bt.setName(newName)
      })
    }
    
    if (Component.current == that.component)
      that.component.addPrePopTask(() => {
        doIt()
      })
    else
      doIt()
  }
}

/**
 * BlackBox wrapper for LINEAR_BLOCK HLS module
 * 
 * LINEAR_BLOCK module interface:
 * - m_axi_in_r: Read-only AXI master for input data (src)
 * - m_axi_out_r: Write-only AXI master for output data (dst)
 * - m_axi_weights: Read-only AXI master for weights (packed 4-bit, bias, scales)
 * - s_axi_control: AXI-Lite slave for control and scalar parameters
 */
class LINEAR_BLOCK_Blackbox extends BlackBox {
  val top_name: String = "LINEAR_BLOCK"
  setDefinitionName(top_name)
  
  private val verilog_file_path: String = s"src/main/verilog/$top_name/all.v"
  
  val io = new Bundle {
    // Clock and reset
    val ap_clk: Bool = in Bool()
    val ap_rst_n: Bool = in Bool()
    
    // AXI master interfaces (multiple bundles)
    val m_axi_in_r: BlackboxAxi = master(BlackboxAxi(BlackboxAxiConfig(verilog_file_path, "IN_R")))
    val m_axi_out_r: BlackboxAxi = master(BlackboxAxi(BlackboxAxiConfig(verilog_file_path, "OUT_R")))
    val m_axi_weights: BlackboxAxi = master(BlackboxAxi(BlackboxAxiConfig(verilog_file_path, "WEIGHTS")))
    
    // AXI-Lite control interfaces
    val s_axi_control: BlackboxAxiLite = slave(BlackboxAxiLite(BlackboxAxiLiteConfig(verilog_file_path)))
    
    // s_axi_control_r interface (address width 7)
    val s_axi_control_r_AWVALID: Bool = in Bool() default(False)
    val s_axi_control_r_AWREADY: Bool = out Bool()
    val s_axi_control_r_AWADDR: UInt = in UInt(7 bits) default(0)
    val s_axi_control_r_WVALID: Bool = in Bool() default(False)
    val s_axi_control_r_WREADY: Bool = out Bool()
    val s_axi_control_r_WDATA: Bits = in Bits(32 bits) default(0)
    val s_axi_control_r_WSTRB: Bits = in Bits(4 bits) default(0)
    val s_axi_control_r_ARVALID: Bool = in Bool() default(False)
    val s_axi_control_r_ARREADY: Bool = out Bool()
    val s_axi_control_r_ARADDR: UInt = in UInt(7 bits) default(0)
    val s_axi_control_r_RVALID: Bool = out Bool()
    val s_axi_control_r_RREADY: Bool = in Bool() default(True)
    val s_axi_control_r_RDATA: Bits = out Bits(32 bits)
    val s_axi_control_r_RRESP: Bits = out Bits(2 bits)
    val s_axi_control_r_BVALID: Bool = out Bool()
    val s_axi_control_r_BREADY: Bool = in Bool() default(True)
    val s_axi_control_r_BRESP: Bits = out Bits(2 bits)
    
    // Interrupt signal
    val interrupt: Bool = out Bool()
  }
  
  noIoPrefix()
  
  // Custom renamers for multiple m_axi interfaces
  CustomBlackboxAxiRenamer(io.m_axi_in_r, "m_axi_in_r")
  CustomBlackboxAxiRenamer(io.m_axi_out_r, "m_axi_out_r")
  CustomBlackboxAxiRenamer(io.m_axi_weights, "m_axi_weights")
  
  // Rename AXI-Lite interface
  BlackboxAxiLiteRenamer(io.s_axi_control)
  
  // Tie off interrupt
  io.interrupt := False
  
  mapClockDomain(clock = io.ap_clk, reset = io.ap_rst_n, resetActiveLevel = LOW)
  addRTLPath(verilog_file_path)
}

/**
 * Component wrapper for LINEAR_BLOCK module
 */
class LINEAR_BLOCK extends Component {
  val top_name: String = "LINEAR_BLOCK"
  setDefinitionName(top_name + "_wrapper")
  
  private val black_box = new LINEAR_BLOCK_Blackbox()
  
  val io = new Bundle {
    val signals: DaisyChain[ManagerSignals] = DaisyChain(ManagerSignals())
    
    // AXI master interfaces
    val m_axi_in_r: Axi4 = master(Axi4(black_box.io.m_axi_in_r.config.to_std_config()))
    val m_axi_out_r: Axi4 = master(Axi4(black_box.io.m_axi_out_r.config.to_std_config()))
    val m_axi_weights: Axi4 = master(Axi4(black_box.io.m_axi_weights.config.to_std_config()))
    
    // AXI-Lite control interfaces
    val s_axi_control: AxiLite4 = slave(AxiLite4(black_box.io.s_axi_control.config.to_std_config()))
    val s_axi_control_r: AxiLite4 = slave(AxiLite4(addressWidth = 7, dataWidth = 32))
    
    // Interrupt signal (renamed to avoid C++ keyword conflict)
    val ap_interrupt: Bool = out Bool()
  }
  
  noIoPrefix()
  
  val manager = new Manager(single = true)
  manager.io.signals <> io.signals
  
  // Connect AXI interfaces
  black_box.io.m_axi_in_r.connect2std(io.m_axi_in_r)
  black_box.io.m_axi_out_r.connect2std(io.m_axi_out_r)
  black_box.io.m_axi_weights.connect2std(io.m_axi_weights)
  black_box.io.s_axi_control.connect2std(io.s_axi_control)
  
  // Connect s_axi_control_r
  io.s_axi_control_r.aw.valid <> black_box.io.s_axi_control_r_AWVALID
  io.s_axi_control_r.aw.ready <> black_box.io.s_axi_control_r_AWREADY
  io.s_axi_control_r.aw.payload.addr <> black_box.io.s_axi_control_r_AWADDR
  io.s_axi_control_r.w.valid <> black_box.io.s_axi_control_r_WVALID
  io.s_axi_control_r.w.ready <> black_box.io.s_axi_control_r_WREADY
  io.s_axi_control_r.w.payload.data <> black_box.io.s_axi_control_r_WDATA
  io.s_axi_control_r.w.payload.strb <> black_box.io.s_axi_control_r_WSTRB
  io.s_axi_control_r.ar.valid <> black_box.io.s_axi_control_r_ARVALID
  io.s_axi_control_r.ar.ready <> black_box.io.s_axi_control_r_ARREADY
  io.s_axi_control_r.ar.payload.addr <> black_box.io.s_axi_control_r_ARADDR
  io.s_axi_control_r.r.valid <> black_box.io.s_axi_control_r_RVALID
  io.s_axi_control_r.r.ready <> black_box.io.s_axi_control_r_RREADY
  io.s_axi_control_r.r.payload.data <> black_box.io.s_axi_control_r_RDATA
  io.s_axi_control_r.r.payload.resp <> black_box.io.s_axi_control_r_RRESP
  io.s_axi_control_r.b.valid <> black_box.io.s_axi_control_r_BVALID
  io.s_axi_control_r.b.ready <> black_box.io.s_axi_control_r_BREADY
  io.s_axi_control_r.b.payload.resp <> black_box.io.s_axi_control_r_BRESP
  
  // Connect interrupt signal
  io.ap_interrupt <> black_box.io.interrupt
  
  // Rename for Vivado compatibility
  Axi4SpecRenamer(io.m_axi_in_r)
  Axi4SpecRenamer(io.m_axi_out_r)
  Axi4SpecRenamer(io.m_axi_weights)
  AxiLite4SpecRenamer(io.s_axi_control)
  AxiLite4SpecRenamer(io.s_axi_control_r)
}

/**
 * Simulation object for LINEAR_BLOCK module
 */
object simulate_linear_block extends App {
  
  val spinalConfig: SpinalConfig = SpinalConfig(
    defaultConfigForClockDomains = ClockDomainConfig(
      resetKind = SYNC, resetActiveLevel = LOW
    )
  )
  
  SimConfig
    .withConfig(spinalConfig)
    .withFstWave
    .withWaveDepth(2)
    .allOptimisation
    .withVerilator
    .addSimulatorFlag("--unroll-count 1024")
    .addSimulatorFlag("-j 16")
    .addSimulatorFlag("-O3 --x-assign fast --x-initial fast --noassert")
    .compile(new LINEAR_BLOCK)
    .doSimUntilVoid { dut =>
      // Model parameters
      val D_MODEL = 192
      val D_INNER = 384
      val NUM_PATCHES = 197
      val LAYER_IDX = 0
      
      // Data path
      val data_path_prefix = utils.DataPathConfig.data_path_prefix
      
      // Create AXI memory simulators
      val axi_mem_in = AxiMemorySim(dut.io.m_axi_in_r, dut.clockDomain, AxiMemorySimConfig())
      val axi_mem_out = AxiMemorySim(dut.io.m_axi_out_r, dut.clockDomain, AxiMemorySimConfig())
      val axi_mem_weights = AxiMemorySim(dut.io.m_axi_weights, dut.clockDomain, AxiMemorySimConfig())
      
      // Create AXI-Lite drivers for control interfaces
      val axilite_driver = AddressSeparableAxiLite4Driver(dut.io.s_axi_control, dut.clockDomain)
      val axilite_driver_r = AddressSeparableAxiLite4Driver(dut.io.s_axi_control_r, dut.clockDomain)
      
      // Memory addresses
      val addr_src = 0x100000L
      val addr_dst = 0x200000L
      val addr_weights_packed = 0x300000L
      val addr_bias = addr_weights_packed + 0x200000L
      val addr_scales = addr_weights_packed + 0x300000L
      
      // AXI-Lite register addresses (standard HLS mapping)
      // s_axi_control registers (address width 6)
      val ADDR_AP_CTRL = 0x00
      val ADDR_OUT_DIM = 0x10
      val ADDR_IN_DIM = 0x18
      val ADDR_NUM_PATCHES = 0x20
      val ADDR_FLAGS = 0x28
      
      // Initialize clock
      init_clock(dut.clockDomain, 10)
      dut.clockDomain.waitSampling(10)
      
      println("=== LINEAR_BLOCK Simulation Test ===")
      
      // Load input data
      println("Loading input data...")
      val input_file = s"$data_path_prefix/ref_float32_block/layers.${LAYER_IDX}.mixer.in_proj.input.float32.bin"
      val input_float = read_float_file(input_file, NUM_PATCHES * D_MODEL)
      val input_data = input_float.map(f => 
        FixedPointTypes.floatToFixed(f, FixedPointTypes.fm_t).toLong
      )
      
      // Load weights (packed 4-bit format, separate X and Z files)
      println("Loading weights...")
      val weight_file_x = s"$data_path_prefix/bin_float32_block/layers.${LAYER_IDX}.mixer.in_proj.weight.x.int4.bin"
      val weight_file_z = s"$data_path_prefix/bin_float32_block/layers.${LAYER_IDX}.mixer.in_proj.weight.z.int4.bin"
      val scale_file_x = s"$data_path_prefix/bin_float32_block/layers.${LAYER_IDX}.mixer.in_proj.weight_scale.x.float32.bin"
      val scale_file_z = s"$data_path_prefix/bin_float32_block/layers.${LAYER_IDX}.mixer.in_proj.weight_scale.z.float32.bin"
      
      val total_weight_elems_used = D_INNER * D_MODEL
      
      val weights_int4 = read_int4_file(weight_file_x, total_weight_elems_used)
      val weights_int4_z = read_int4_file(weight_file_z, total_weight_elems_used)
      
      println(s"Loaded ${weights_int4.length} X weight elements and ${weights_int4_z.length} Z weight elements")
      
      // Load weight scales (per-block format: [out_dim][num_in_blocks])
      // WEIGHT_BLOCK_SIZE = 32, so num_in_blocks = ceildiv(192, 32) = 6
      // Total scales = D_INNER * num_in_blocks = 384 * 6 = 2304
      val WEIGHT_BLOCK_SIZE = 32
      val num_in_blocks = (D_MODEL + WEIGHT_BLOCK_SIZE - 1) / WEIGHT_BLOCK_SIZE  // 192/32 = 6
      val scales_per_proj = D_INNER * num_in_blocks  // 384 * 6 = 2304
      
      val weight_scales_float = read_float_file(scale_file_x, scales_per_proj)
      val weight_scales_fixed = weight_scales_float.map(f => 
        FixedPointTypes.floatToFixed(f, FixedPointTypes.wt_linear_ws_t).toLong
      )
      val weight_scales_float_z = read_float_file(scale_file_z, scales_per_proj)
      val weight_scales_fixed_z = weight_scales_float_z.map(f => 
        FixedPointTypes.floatToFixed(f, FixedPointTypes.wt_linear_ws_t).toLong
      )
      
      println(s"Loaded ${weight_scales_fixed.length} weight scales (per-block: ${num_in_blocks} blocks)")
      
      // Bias is zero
      val bias_data = Array.fill[Long](D_INNER)(0L)
      
      // Load reference outputs
      println("Loading reference output...")
      val ref_file = s"$data_path_prefix/ref_float32_block/layers.${LAYER_IDX}.mixer.in_proj.output.x.float32.bin"
      val ref_output_float = read_float_file(ref_file, NUM_PATCHES * D_INNER)
      val ref_output = ref_output_float.map(f => 
        FixedPointTypes.floatToFixed(f, FixedPointTypes.fm_t).toLong
      )
      
      // Write data to AXI memory
      println("Writing data to AXI memory...")
      val AXI_XFER_BIT_WIDTH_VAL = 256
      val FEATURE_BLOCK_SIZE = 8
      val NUM_FEATURE_BLOCKS = (D_MODEL + FEATURE_BLOCK_SIZE - 1) / FEATURE_BLOCK_SIZE
      val total_blocks = NUM_PATCHES * NUM_FEATURE_BLOCKS
      
      println(s"Writing input data: $NUM_PATCHES patches, $NUM_FEATURE_BLOCKS blocks per patch")
      
      // Write input data in fm_block_t format
      for (p <- 0 until NUM_PATCHES) {
        for (b <- 0 until NUM_FEATURE_BLOCKS) {
          val block_idx = p * NUM_FEATURE_BLOCKS + b
          val block_base_addr = (addr_src + block_idx * 32).toLong
          for (o <- 0 until FEATURE_BLOCK_SIZE) {
            val global_idx = p * D_MODEL + b * FEATURE_BLOCK_SIZE + o
            if (global_idx < input_data.length) {
              val value = BigInt(input_data(global_idx))
              val unsigned_value = if (value < 0) {
                (BigInt(1) << 32) + value
              } else {
                value
              }
              val elem_addr = block_base_addr + o * 4
              axi_mem_in.memory.writeBigInt(elem_addr, unsigned_value, 4)
            }
          }
        }
      }
      
      // Write packed weights
      write_int4_array_to_axi(axi_mem_weights, dut.clockDomain, addr_weights_packed, weights_int4, total_weight_elems_used, AXI_XFER_BIT_WIDTH_VAL, "Packed Weights")
      
      // Write bias and scales (256-bit words, 8 elements per word)
      val BIAS_PER_WORD = AXI_XFER_BIT_WIDTH_VAL / 32
      val bias_words = (D_INNER + BIAS_PER_WORD - 1) / BIAS_PER_WORD
      for (w <- 0 until bias_words) {
        val word_base_addr = (addr_bias + w * (AXI_XFER_BIT_WIDTH_VAL / 8)).toLong
        for (j <- 0 until BIAS_PER_WORD) {
          val idx = w * BIAS_PER_WORD + j
          if (idx < D_INNER) {
            val value = BigInt(bias_data(idx))
            val unsigned_value = if (value < 0) {
              (BigInt(1) << 32) + value
            } else {
              value
            }
            val elem_addr = word_base_addr + j * 4
            axi_mem_weights.memory.writeBigInt(elem_addr, unsigned_value, 4)
          }
        }
      }
      
      // Write scales (per-block format: scales_per_proj elements)
      val SCALES_PER_WORD = AXI_XFER_BIT_WIDTH_VAL / 32  // 8 float32 per 256-bit word
      val scale_words = (scales_per_proj + SCALES_PER_WORD - 1) / SCALES_PER_WORD
      for (w <- 0 until scale_words) {
        val word_base_addr = (addr_scales + w * (AXI_XFER_BIT_WIDTH_VAL / 8)).toLong
        for (j <- 0 until SCALES_PER_WORD) {
          val idx = w * SCALES_PER_WORD + j
          if (idx < scales_per_proj) {
            val value = BigInt(weight_scales_fixed(idx))
            val unsigned_value = if (value < 0) {
              (BigInt(1) << 32) + value
            } else {
              value
            }
            val elem_addr = word_base_addr + j * 4
            axi_mem_weights.memory.writeBigInt(elem_addr, unsigned_value, 4)
          }
        }
      }
      
      // Reset AXI memory simulators
      axi_mem_in.reset()
      axi_mem_out.reset()
      axi_mem_weights.reset()
      axilite_driver.reset()
      axilite_driver_r.reset()
      
      println("Waiting after reset for interfaces to stabilize...")
      dut.clockDomain.waitSampling(100)
      
      // Configure module via AXI-Lite
      println("Configuring module via AXI-Lite...")
      
      // Write memory addresses to s_axi_control_r
      println("Writing memory addresses to s_axi_control_r...")
      axilite_driver_r.write(0x10, (addr_dst & 0xFFFFFFFFL).toInt)
      axilite_driver_r.write(0x14, ((addr_dst >> 32) & 0xFFFFFFFFL).toInt)
      axilite_driver_r.write(0x1c, (addr_src & 0xFFFFFFFFL).toInt)
      axilite_driver_r.write(0x20, ((addr_src >> 32) & 0xFFFFFFFFL).toInt)
      axilite_driver_r.write(0x28, (addr_weights_packed & 0xFFFFFFFFL).toInt)
      axilite_driver_r.write(0x2c, ((addr_weights_packed >> 32) & 0xFFFFFFFFL).toInt)
      axilite_driver_r.write(0x34, (addr_scales & 0xFFFFFFFFL).toInt)
      axilite_driver_r.write(0x38, ((addr_scales >> 32) & 0xFFFFFFFFL).toInt)
      axilite_driver_r.write(0x40, (addr_bias & 0xFFFFFFFFL).toInt)
      axilite_driver_r.write(0x44, ((addr_bias >> 32) & 0xFFFFFFFFL).toInt)
      
      // Write scalar parameters to s_axi_control
      println("Writing scalar parameters to s_axi_control...")
      axilite_driver.write(ADDR_OUT_DIM, D_INNER)
      axilite_driver.write(ADDR_IN_DIM, D_MODEL)
      axilite_driver.write(ADDR_NUM_PATCHES, NUM_PATCHES)
      axilite_driver.write(ADDR_FLAGS, 0) // No bias, no SiLU
      
      println("Configuration complete")
      dut.clockDomain.waitSampling(10)
      
      // Start computation
      println("Starting computation...")
      axilite_driver.write(ADDR_AP_CTRL, 0x11) // AP_START=1, AP_CONTINUE=1
      dut.clockDomain.waitSampling()
      axilite_driver.write(ADDR_AP_CTRL, 0x10) // Clear AP_START, keep AP_CONTINUE=1
      
      // Wait for completion (poll AP_DONE)
      println("Waiting for completion...")
      var done = false
      var cycles = 0
      val max_cycles = 10000000
      while (!done && cycles < max_cycles) {
        val ctrl = axilite_driver.read(ADDR_AP_CTRL)
        done = (ctrl & 2) != 0 // AP_DONE bit
        if (!done) {
          dut.clockDomain.waitSampling(100)
          cycles += 100
        }
      }
      
      if (!done) {
        println(s"ERROR: Computation did not complete within $max_cycles cycles")
        simFailure()
      } else {
        println(s"Computation completed in $cycles cycles")
      }
      
      // Read results from AXI memory
      println("Reading results from AXI memory...")
      val FEATURE_BLOCK_SIZE_OUT = 8
      val NUM_FEATURE_BLOCKS_OUT = (D_INNER + FEATURE_BLOCK_SIZE_OUT - 1) / FEATURE_BLOCK_SIZE_OUT
      val total_blocks_out = NUM_PATCHES * NUM_FEATURE_BLOCKS_OUT
      val dut_output_flat = new Array[Long](NUM_PATCHES * D_INNER)
      
      for (p <- 0 until NUM_PATCHES) {
        for (b <- 0 until NUM_FEATURE_BLOCKS_OUT) {
          val block_idx = p * NUM_FEATURE_BLOCKS_OUT + b
          val block_base_addr = (addr_dst + block_idx * 32).toLong
          for (o <- 0 until FEATURE_BLOCK_SIZE_OUT) {
            val global_idx = p * D_INNER + b * FEATURE_BLOCK_SIZE_OUT + o
            if (global_idx < dut_output_flat.length) {
              val elem_addr = block_base_addr + o * 4
              val unsigned_value = axi_mem_out.memory.readBigInt(elem_addr, 4)
              val signed_value = if (unsigned_value >= (BigInt(1) << 31)) {
                unsigned_value - (BigInt(1) << 32)
              } else {
                unsigned_value
              }
              dut_output_flat(global_idx) = signed_value.toLong
            }
          }
        }
      }
      
      val dut_output = dut_output_flat
      
      // Compare results
      println("Comparing results...")
      compare_arrays(ref_output, dut_output, "LINEAR(in_proj_x) Output", FixedPointTypes.fm_t)
      
      println("\n" + "=" * 80)
      println("=== Testing LINEAR_BLOCK(in_proj_z) with FLAG_SILU ===")
      println("=" * 80)
      
      // Test in_proj_z: same input, but Z weights and FLAG_SILU (already loaded above)
      println(s"Using Z weights: ${weights_int4_z.length} elements")
      println(s"Using Z scales: ${weight_scales_fixed_z.length} scales")
      
      // Write Z weights to memory
      val addr_weights_packed_z = 0x700000L
      val addr_scales_z = 0x800000L
      val addr_dst_z = 0x900000L
      
      println("Writing Z weights to memory...")
      write_int4_array_to_axi(axi_mem_weights, dut.clockDomain, addr_weights_packed_z, weights_int4_z, total_weight_elems_used, AXI_XFER_BIT_WIDTH_VAL, "Z Packed Weights")
      
      // Write Z scales (per-block format)
      for (w <- 0 until scale_words) {
        val word_base_addr = (addr_scales_z + w * (AXI_XFER_BIT_WIDTH_VAL / 8)).toLong
        for (j <- 0 until SCALES_PER_WORD) {
          val idx = w * SCALES_PER_WORD + j
          if (idx < scales_per_proj) {
            val value = BigInt(weight_scales_fixed_z(idx))
            val unsigned_value = if (value < 0) {
              (BigInt(1) << 32) + value
            } else {
              value
            }
            val elem_addr = word_base_addr + j * 4
            axi_mem_weights.memory.writeBigInt(elem_addr, unsigned_value, 4)
          }
        }
      }
      
      // Reset AXI memory simulators
      axi_mem_in.reset()
      axi_mem_out.reset()
      axi_mem_weights.reset()
      axilite_driver.reset()
      axilite_driver_r.reset()
      
      println("Waiting after reset for interfaces to stabilize...")
      dut.clockDomain.waitSampling(100)
      
      // Write memory addresses to s_axi_control_r
      println("Writing memory addresses to s_axi_control_r...")
      axilite_driver_r.write(0x10, (addr_dst_z & 0xFFFFFFFFL).toInt)
      axilite_driver_r.write(0x14, ((addr_dst_z >> 32) & 0xFFFFFFFFL).toInt)
      axilite_driver_r.write(0x1c, (addr_src & 0xFFFFFFFFL).toInt)
      axilite_driver_r.write(0x20, ((addr_src >> 32) & 0xFFFFFFFFL).toInt)
      axilite_driver_r.write(0x28, (addr_weights_packed_z & 0xFFFFFFFFL).toInt)
      axilite_driver_r.write(0x2c, ((addr_weights_packed_z >> 32) & 0xFFFFFFFFL).toInt)
      axilite_driver_r.write(0x34, (addr_scales_z & 0xFFFFFFFFL).toInt)
      axilite_driver_r.write(0x38, ((addr_scales_z >> 32) & 0xFFFFFFFFL).toInt)
      axilite_driver_r.write(0x40, (addr_bias & 0xFFFFFFFFL).toInt)
      axilite_driver_r.write(0x44, ((addr_bias >> 32) & 0xFFFFFFFFL).toInt)
      
      // Write scalar parameters to s_axi_control
      println("Writing scalar parameters to s_axi_control...")
      axilite_driver.write(ADDR_OUT_DIM, D_INNER)
      axilite_driver.write(ADDR_IN_DIM, D_MODEL)
      axilite_driver.write(ADDR_NUM_PATCHES, NUM_PATCHES)
      axilite_driver.write(ADDR_FLAGS, 2) // FLAG_SILU = 2
      
      println("Configuration complete")
      dut.clockDomain.waitSampling(10)
      
      // Start computation
      println("Starting computation...")
      axilite_driver.write(ADDR_AP_CTRL, 0x11) // AP_START=1, AP_CONTINUE=1
      dut.clockDomain.waitSampling()
      axilite_driver.write(ADDR_AP_CTRL, 0x10) // Clear AP_START, keep AP_CONTINUE=1
      
      // Wait for completion
      var done_z = false
      var cycles_z = 0
      val max_cycles_z = 10000000
      while (!done_z && cycles_z < max_cycles_z) {
        val ctrl = axilite_driver.read(ADDR_AP_CTRL)
        done_z = (ctrl & 2) != 0
        if (!done_z) {
          dut.clockDomain.waitSampling(100)
          cycles_z += 100
        }
      }
      
      if (!done_z) {
        println(s"ERROR: Computation did not complete within $max_cycles_z cycles")
        simFailure()
      } else {
        println(s"Computation completed in $cycles_z cycles")
      }
      
      // Read results from AXI memory
      println("Reading results from AXI memory...")
      val dut_output_z = new Array[Long](NUM_PATCHES * D_INNER)
      for (p <- 0 until NUM_PATCHES) {
        for (b <- 0 until NUM_FEATURE_BLOCKS_OUT) {
          val block_idx = p * NUM_FEATURE_BLOCKS_OUT + b
          val block_base_addr = (addr_dst_z + block_idx * 32).toLong
          for (o <- 0 until FEATURE_BLOCK_SIZE_OUT) {
            val global_idx = p * D_INNER + b * FEATURE_BLOCK_SIZE_OUT + o
            if (global_idx < dut_output_z.length) {
              val elem_addr = block_base_addr + o * 4
              val unsigned_value = axi_mem_out.memory.readBigInt(elem_addr, 4)
              val signed_value = if (unsigned_value >= (BigInt(1) << 31)) {
                unsigned_value - (BigInt(1) << 32)
              } else {
                unsigned_value
              }
              dut_output_z(global_idx) = signed_value.toLong
            }
          }
        }
      }
      
      // Compare results
      println("Comparing results...")
      val ref_file_z = s"$data_path_prefix/ref_float32_block/layers.${LAYER_IDX}.mixer.in_proj.output.z_silu.float32.bin"
      val ref_output_float_z = read_float_file(ref_file_z, NUM_PATCHES * D_INNER)
      val ref_output_z = ref_output_float_z.map(f => 
        FixedPointTypes.floatToFixed(f, FixedPointTypes.fm_t).toLong
      )
      compare_arrays(ref_output_z, dut_output_z, "LINEAR(in_proj_z) Output", FixedPointTypes.fm_t)
      
      println("\n" + "=" * 80)
      println("=== Testing LINEAR_BLOCK(head) ===")
      println("=" * 80)
      
      // Model parameters for HEAD
      val D_MODEL_HEAD = 192
      val PADDED_NUM_CLASSES = 1008
      val NUM_PATCHES_HEAD = 1
      
      // Memory addresses for HEAD
      val addr_src_head = 0x1000000L
      val addr_dst_head = 0x1100000L
      val addr_weights_packed_head = 0x1200000L
      val addr_bias_head = addr_weights_packed_head + 0x200000L
      val addr_scales_head = addr_weights_packed_head + 0x300000L
      
      // Load HEAD input data
      println("Loading HEAD input data...")
      val input_file_head = s"$data_path_prefix/ref_float32_block/head.input.float32.bin"
      val input_float_head = read_float_file(input_file_head, NUM_PATCHES_HEAD * D_MODEL_HEAD)
      val input_data_head = input_float_head.map(f => 
        FixedPointTypes.floatToFixed(f, FixedPointTypes.fm_t).toLong
      )
      
      // Load HEAD weights
      println("Loading HEAD weights...")
      val weight_file_head = s"$data_path_prefix/bin_float32_block/head.weight.int4.bin"
      val scale_file_head = s"$data_path_prefix/bin_float32_block/head.weight_scale.float32.bin"
      val bias_file_head = s"$data_path_prefix/bin_float32_block/head.bias.float32.bin"
      
      val total_weight_elems_head = PADDED_NUM_CLASSES * D_MODEL_HEAD
      val weights_int4_head = read_int4_file(weight_file_head, total_weight_elems_head)
      
      // Load HEAD weight scales
      val num_in_blocks_head = (D_MODEL_HEAD + WEIGHT_BLOCK_SIZE - 1) / WEIGHT_BLOCK_SIZE  // 192/32 = 6
      val weight_scales_float_head_raw = read_float_file(scale_file_head, PADDED_NUM_CLASSES)
      val weight_scales_float_head = new Array[Float](PADDED_NUM_CLASSES * num_in_blocks_head)
      for (i <- 0 until PADDED_NUM_CLASSES) {
        for (j <- 0 until num_in_blocks_head) {
          weight_scales_float_head(i * num_in_blocks_head + j) = weight_scales_float_head_raw(i)
        }
      }
      val weight_scales_fixed_head = weight_scales_float_head.map(f => 
        FixedPointTypes.floatToFixed(f, FixedPointTypes.wt_linear_ws_t).toLong
      )
      
      // Load HEAD bias
      println("Loading HEAD bias...")
      val bias_float_head = read_float_file(bias_file_head, PADDED_NUM_CLASSES)
      val bias_data_head = bias_float_head.map(f => 
        FixedPointTypes.floatToFixed(f, FixedPointTypes.wt_linear_bias_t).toLong
      )
      
      // Write HEAD data to AXI memory
      println("Writing HEAD data to AXI memory...")
      val NUM_FEATURE_BLOCKS_HEAD = (D_MODEL_HEAD + FEATURE_BLOCK_SIZE - 1) / FEATURE_BLOCK_SIZE
      
      for (p <- 0 until NUM_PATCHES_HEAD) {
        for (b <- 0 until NUM_FEATURE_BLOCKS_HEAD) {
          val block_idx = p * NUM_FEATURE_BLOCKS_HEAD + b
          val block_base_addr = (addr_src_head + block_idx * 32).toLong
          for (o <- 0 until FEATURE_BLOCK_SIZE) {
            val global_idx = p * D_MODEL_HEAD + b * FEATURE_BLOCK_SIZE + o
            if (global_idx < input_data_head.length) {
              val value = BigInt(input_data_head(global_idx))
              val unsigned_value = if (value < 0) (BigInt(1) << 32) + value else value
              val elem_addr = block_base_addr + o * 4
              axi_mem_in.memory.writeBigInt(elem_addr, unsigned_value, 4)
            }
          }
        }
      }
      
      write_int4_array_to_axi(axi_mem_weights, dut.clockDomain, addr_weights_packed_head, weights_int4_head, total_weight_elems_head, AXI_XFER_BIT_WIDTH_VAL, "HEAD Packed Weights")
      
      // Write HEAD bias
      val bias_words_head = (PADDED_NUM_CLASSES + BIAS_PER_WORD - 1) / BIAS_PER_WORD
      for (w <- 0 until bias_words_head) {
        val word_base_addr = (addr_bias_head + w * (AXI_XFER_BIT_WIDTH_VAL / 8)).toLong
        for (j <- 0 until BIAS_PER_WORD) {
          val idx = w * BIAS_PER_WORD + j
          if (idx < PADDED_NUM_CLASSES) {
            val value = BigInt(bias_data_head(idx))
            val unsigned_value = if (value < 0) (BigInt(1) << 32) + value else value
            val elem_addr = word_base_addr + j * 4
            axi_mem_weights.memory.writeBigInt(elem_addr, unsigned_value, 4)
          }
        }
      }
      
      // Write HEAD scales
      val total_scales_head = weight_scales_fixed_head.length
      val scale_words_head = (total_scales_head + SCALES_PER_WORD - 1) / SCALES_PER_WORD
      for (w <- 0 until scale_words_head) {
        val word_base_addr = (addr_scales_head + w * (AXI_XFER_BIT_WIDTH_VAL / 8)).toLong
        for (j <- 0 until SCALES_PER_WORD) {
          val idx = w * SCALES_PER_WORD + j
          if (idx < total_scales_head) {
            val value = BigInt(weight_scales_fixed_head(idx))
            val unsigned_value = if (value < 0) (BigInt(1) << 32) + value else value
            val elem_addr = word_base_addr + j * 4
            axi_mem_weights.memory.writeBigInt(elem_addr, unsigned_value, 4)
          }
        }
      }
      
      // Reset and configure for HEAD
      axi_mem_in.reset(); axi_mem_out.reset(); axi_mem_weights.reset(); axilite_driver.reset(); axilite_driver_r.reset()
      dut.clockDomain.waitSampling(100)
      
      axilite_driver_r.write(0x10, (addr_dst_head & 0xFFFFFFFFL).toInt)
      axilite_driver_r.write(0x14, ((addr_dst_head >> 32) & 0xFFFFFFFFL).toInt)
      axilite_driver_r.write(0x1c, (addr_src_head & 0xFFFFFFFFL).toInt)
      axilite_driver_r.write(0x20, ((addr_src_head >> 32) & 0xFFFFFFFFL).toInt)
      axilite_driver_r.write(0x28, (addr_weights_packed_head & 0xFFFFFFFFL).toInt)
      axilite_driver_r.write(0x2c, ((addr_weights_packed_head >> 32) & 0xFFFFFFFFL).toInt)
      axilite_driver_r.write(0x34, (addr_scales_head & 0xFFFFFFFFL).toInt)
      axilite_driver_r.write(0x38, ((addr_scales_head >> 32) & 0xFFFFFFFFL).toInt)
      axilite_driver_r.write(0x40, (addr_bias_head & 0xFFFFFFFFL).toInt)
      axilite_driver_r.write(0x44, ((addr_bias_head >> 32) & 0xFFFFFFFFL).toInt)
      
      axilite_driver.write(ADDR_OUT_DIM, PADDED_NUM_CLASSES)
      axilite_driver.write(ADDR_IN_DIM, D_MODEL_HEAD)
      axilite_driver.write(ADDR_NUM_PATCHES, NUM_PATCHES_HEAD)
      axilite_driver.write(ADDR_FLAGS, 1) // FLAG_BIAS = 1
      
      println("Starting HEAD computation...")
      axilite_driver.write(ADDR_AP_CTRL, 0x11)
      dut.clockDomain.waitSampling()
      axilite_driver.write(ADDR_AP_CTRL, 0x10)
      
      var done_head = false; var cycles_head = 0
      while (!done_head && cycles_head < 10000000) {
        val ctrl = axilite_driver.read(ADDR_AP_CTRL)
        done_head = (ctrl & 2) != 0
        if (!done_head) { dut.clockDomain.waitSampling(100); cycles_head += 100 }
      }
      
      if (!done_head) { println("ERROR: Timeout"); simFailure() } else println(s"Completed in $cycles_head cycles")
      
      // Read HEAD results
      val NUM_FEATURE_BLOCKS_OUT_HEAD = (PADDED_NUM_CLASSES + FEATURE_BLOCK_SIZE_OUT) / FEATURE_BLOCK_SIZE_OUT // Simplified
      val dut_output_head = new Array[Long](NUM_PATCHES_HEAD * PADDED_NUM_CLASSES)
      for (p <- 0 until NUM_PATCHES_HEAD) {
        val num_out_blocks = (PADDED_NUM_CLASSES + FEATURE_BLOCK_SIZE_OUT - 1) / FEATURE_BLOCK_SIZE_OUT
        for (b <- 0 until num_out_blocks) {
          val block_base_addr = (addr_dst_head + (p * num_out_blocks + b) * 32).toLong
          for (o <- 0 until FEATURE_BLOCK_SIZE_OUT) {
            val idx = p * PADDED_NUM_CLASSES + b * FEATURE_BLOCK_SIZE_OUT + o
            if (idx < dut_output_head.length) {
              val unsigned = axi_mem_out.memory.readBigInt(block_base_addr + o * 4, 4)
              dut_output_head(idx) = (if (unsigned >= (BigInt(1) << 31)) unsigned - (BigInt(1) << 32) else unsigned).toLong
            }
          }
        }
      }
      
      // Load HEAD reference outputs
      println("Loading HEAD reference output...")
      val ref_file_head = s"$data_path_prefix/ref_float32_block/head.output.float32.bin"
      val ref_output_float_head = read_float_file(ref_file_head, NUM_PATCHES_HEAD * PADDED_NUM_CLASSES)
      val ref_output_head = ref_output_float_head.map(f => 
        FixedPointTypes.floatToFixed(f, FixedPointTypes.fm_t).toLong
      )
      
      compare_arrays(ref_output_head, dut_output_head, "LINEAR(head) Output", FixedPointTypes.fm_t)
      
      println("\nAll LINEAR_BLOCK tests completed successfully!")
      simSuccess()
      dut.clockDomain.waitSampling(100)
    }
}
