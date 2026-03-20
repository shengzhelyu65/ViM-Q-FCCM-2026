import spinal.core.sim._
import spinal.core._
import spinal.lib.bus.amba4.axi.sim.{AxiMemorySim, AxiMemorySimConfig}
import spinal.lib.bus.amba4.axi._
import spinal.lib.bus.amba4.axilite.{AxiLite4, AxiLite4SpecRenamer}
import spinal.lib._
import utils._

import scala.language.postfixOps

class SMOOTH_Blackbox extends BlackBox {
  val top_name: String = "SMOOTH"
  setDefinitionName(top_name)
  
  private val verilog_file_path: String = s"src/main/verilog/$top_name/all.v"
  
  val io = new Bundle {
    // Clock and reset
    val ap_clk: Bool = in Bool()
    val ap_rst_n: Bool = in Bool()
    
    // AXI master interfaces
    val m_axi_in_r: BlackboxAxi = master(BlackboxAxi(BlackboxAxiConfig(verilog_file_path, "IN_R")))
    val m_axi_out_r: BlackboxAxi = master(BlackboxAxi(BlackboxAxiConfig(verilog_file_path, "OUT_R")))
    val m_axi_weights: BlackboxAxi = master(BlackboxAxi(BlackboxAxiConfig(verilog_file_path, "WEIGHTS")))
    
    // AXI-Lite control interfaces
    val s_axi_control: BlackboxAxiLite = slave(BlackboxAxiLite(BlackboxAxiLiteConfig(verilog_file_path)))
    
    // s_axi_control_r interface - create individual signals with correct names
    // Address width is 6 (from synthesis report)
    val s_axi_control_r_AWVALID: Bool = in Bool() default(False)
    val s_axi_control_r_AWREADY: Bool = out Bool()
    val s_axi_control_r_AWADDR: UInt = in UInt(6 bits) default(0)
    val s_axi_control_r_WVALID: Bool = in Bool() default(False)
    val s_axi_control_r_WREADY: Bool = out Bool()
    val s_axi_control_r_WDATA: Bits = in Bits(32 bits) default(0)
    val s_axi_control_r_WSTRB: Bits = in Bits(4 bits) default(0)
    val s_axi_control_r_ARVALID: Bool = in Bool() default(False)
    val s_axi_control_r_ARREADY: Bool = out Bool()
    val s_axi_control_r_ARADDR: UInt = in UInt(6 bits) default(0)
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
  
  // Tie off interrupt (not used in simulation)
  io.interrupt := False
  
  mapClockDomain(clock = io.ap_clk, reset = io.ap_rst_n, resetActiveLevel = LOW)
  addRTLPath(verilog_file_path)
}

/**
 * SMOOTH Component wrapper
 */
class SMOOTH extends Component {
  val top_name: String = "SMOOTH"
  setDefinitionName(top_name + "_wrapper")
  
  private val black_box = new SMOOTH_Blackbox()
  
  val io = new Bundle {
    val signals: DaisyChain[ManagerSignals] = DaisyChain(ManagerSignals())
    
    // AXI master interfaces
    val m_axi_in_r: Axi4 = master(Axi4(black_box.io.m_axi_in_r.config.to_std_config()))
    val m_axi_out_r: Axi4 = master(Axi4(black_box.io.m_axi_out_r.config.to_std_config()))
    val m_axi_weights: Axi4 = master(Axi4(black_box.io.m_axi_weights.config.to_std_config()))
    
    // AXI-Lite control interfaces
    val s_axi_control: AxiLite4 = slave(AxiLite4(black_box.io.s_axi_control.config.to_std_config()))
    // s_axi_control_r has address width 6 (from synthesis report)
    val s_axi_control_r: AxiLite4 = slave(AxiLite4(addressWidth = 6, dataWidth = 32))
  }
  
  noIoPrefix()
  
  // Manager for control signals (if needed)
  val manager = new Manager(single = true)
  manager.io.signals <> io.signals
  
  // Connect AXI interfaces
  black_box.io.m_axi_in_r.connect2std(io.m_axi_in_r)
  black_box.io.m_axi_out_r.connect2std(io.m_axi_out_r)
  black_box.io.m_axi_weights.connect2std(io.m_axi_weights)
  black_box.io.s_axi_control.connect2std(io.s_axi_control)
  
  // Connect s_axi_control_r (individual signals to AxiLite4 bundle)
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
  io.s_axi_control_r.b.resp <> black_box.io.s_axi_control_r_BRESP
  
  // Interrupt is tied off in blackbox, not exposed in wrapper (like LINEAR)
  
  // Rename AXI-Lite for Vivado compatibility
  AxiLite4SpecRenamer(io.s_axi_control)
  AxiLite4SpecRenamer(io.s_axi_control_r)
}

/**
 * Simulation function for SMOOTH module
 */
object simulate_smooth extends App {
    val spinalConfig = SpinalConfig(
      targetDirectory = "tmp/",
      defaultConfigForClockDomains = ClockDomainConfig(resetKind = SYNC, resetActiveLevel = LOW)
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
      .addSimulatorFlag("-Wno-SYMRSVDWORD")
      .workspacePath("simWorkspace/SMOOTH_wrapper")
      .compile(new SMOOTH)
      .doSimUntilVoid { dut =>
        println("=== SMOOTH Simulation Test ===")
        
        // Constants
        val D_INNER = 384
        val NUM_PATCHES = 197
        val LAYER_IDX = 0
        val FEATURE_BLOCK_SIZE = 8
        val NUM_FEATURE_BLOCKS_INNER = (D_INNER + FEATURE_BLOCK_SIZE - 1) / FEATURE_BLOCK_SIZE // ceildiv(384, 8) = 48
        
        // Data path
        val data_path_prefix = utils.DataPathConfig.data_path_prefix
        
        // Memory addresses (reduced to avoid page indexing issues)
        val addr_src = 0x100000L
        val addr_dst = 0x200000L
        val addr_smooth_scales = 0x300000L
        
        // Initialize clock
        init_clock(dut.clockDomain, 10)
        dut.clockDomain.waitSampling(10)
        
        // Load input data (float32 -> fixed-point)
        // Use FixedPointTypes.fm_t for input/output data (ap_fixed<32, 14>)
        println("Loading input data...")
        val input_file = s"$data_path_prefix/ref_float32_block/layers.${LAYER_IDX}.mixer.x_proj.input.float32.bin"
        val input_float = read_float_file(input_file, NUM_PATCHES * D_INNER)
        val input_data = input_float.map(f => 
          FixedPointTypes.floatToFixed(f, FixedPointTypes.fm_t).toLong
        )
        
        // Load smooth scales
        // Use FixedPointTypes to ensure correct conversion for wt_linear_ss_t (ap_ufixed<32, 6>)
        println("Loading smooth scales...")
        val scale_file = s"$data_path_prefix/bin_float32_block/layers.${LAYER_IDX}.mixer.x_proj.smooth_scales.float32.bin"
        val smooth_scales_float = read_float_file(scale_file, D_INNER)
        // Convert float32 to ap_ufixed<32, 6> using the type definition
        val smooth_scales_fixed = smooth_scales_float.map(f => 
          FixedPointTypes.floatToFixed(f, FixedPointTypes.wt_linear_ss_t).toLong
        )
        
        // Load reference output (float32 -> fixed-point)
        // Use FixedPointTypes.fm_t for output data (ap_fixed<32, 14>)
        println("Loading reference output...")
        // Reference is input * smooth_scales (computed in C++ testbench)
        val ref_output_float = new Array[Float](NUM_PATCHES * D_INNER)
        for (p <- 0 until NUM_PATCHES) {
          for (d <- 0 until D_INNER) {
            val input_idx = p * D_INNER + d
            ref_output_float(input_idx) = input_float(input_idx) * smooth_scales_float(d)
          }
        }
        val ref_output = ref_output_float.map(f => 
          FixedPointTypes.floatToFixed(f, FixedPointTypes.fm_t).toLong
        )
        
        // Initialize AXI memory simulators FIRST
        val axi_mem_in = AxiMemorySim(dut.io.m_axi_in_r, dut.clockDomain, AxiMemorySimConfig())
        val axi_mem_out = AxiMemorySim(dut.io.m_axi_out_r, dut.clockDomain, AxiMemorySimConfig())
        val axi_mem_weights = AxiMemorySim(dut.io.m_axi_weights, dut.clockDomain, AxiMemorySimConfig())
        
        // Initialize AXI-Lite drivers
        val axilite_driver = AddressSeparableAxiLite4Driver(dut.io.s_axi_control, dut.clockDomain)
        val axilite_driver_r = AddressSeparableAxiLite4Driver(dut.io.s_axi_control_r, dut.clockDomain)
        
        // Write data to AXI memory
        println("Writing data to AXI memory...")
        val AXI_XFER_BIT_WIDTH_VAL = 256
        
        // Write input data as 256-bit words (fm_block_t format)
        // CRITICAL: Write each 32-bit element individually to ensure correct byte ordering
        // AxiMemorySim.writeBigInt writes in LITTLE-ENDIAN byte order
        // HLS expects element 0 at lowest address, element 1 at address+4, etc.
        for (p <- 0 until NUM_PATCHES) {
          for (b <- 0 until NUM_FEATURE_BLOCKS_INNER) {
            val block_idx = p * NUM_FEATURE_BLOCKS_INNER + b
            val block_base_addr = (addr_src + block_idx * 32).toLong
            // Write each 32-bit element at its byte offset (little-endian)
            for (o <- 0 until FEATURE_BLOCK_SIZE) {
              val global_idx = p * D_INNER + b * FEATURE_BLOCK_SIZE + o
              if (global_idx < input_data.length) {
                val value = BigInt(input_data(global_idx))
                val unsigned_value = if (value < 0) {
                  (BigInt(1) << 32) + value
                } else {
                  value
                }
                // Write 32-bit element at byte offset o*4 (little-endian byte order)
                val elem_addr = block_base_addr + o * 4
                axi_mem_in.memory.writeBigInt(elem_addr, unsigned_value, 4)
              }
            }
          }
        }
        
        // Write smooth scales (256-bit words, 8 elements per word)
        // Write each 32-bit element individually to ensure correct byte ordering
        val SCALES_PER_WORD = AXI_XFER_BIT_WIDTH_VAL / 32 // 8 elements per 256-bit word
        val scale_words = (D_INNER + SCALES_PER_WORD - 1) / SCALES_PER_WORD
        for (w <- 0 until scale_words) {
          val word_base_addr = (addr_smooth_scales + w * 32).toLong
          for (j <- 0 until SCALES_PER_WORD) {
            val idx = w * SCALES_PER_WORD + j
            if (idx < D_INNER) {
              val value = BigInt(smooth_scales_fixed(idx))
              val unsigned_value = if (value < 0) {
                (BigInt(1) << 32) + value
              } else {
                value
              }
              // Write 32-bit element at byte offset j*4 (little-endian byte order)
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
        
        // Wait after reset for interfaces to stabilize
        println("Waiting after reset for interfaces to stabilize...")
        dut.clockDomain.waitSampling(100)
        
        // Configure module via AXI-Lite
        println("Configuring module via AXI-Lite...")
        
        // Write memory addresses to s_axi_control_r FIRST
        // From synthesis report: s_axi_control_r has address width 6
        // ADDR_DST_DATA_0 = 6'h10, ADDR_DST_DATA_1 = 6'h14
        // ADDR_SRC_DATA_0 = 6'h1c, ADDR_SRC_DATA_1 = 6'h20
        // ADDR_SMOOTH_SCALES_BASE_DATA_0 = 6'h28, ADDR_SMOOTH_SCALES_BASE_DATA_1 = 6'h2c
        println("Writing memory addresses to s_axi_control_r...")
        axilite_driver_r.write(0x10, (addr_dst & 0xFFFFFFFFL).toInt)
        axilite_driver_r.write(0x14, ((addr_dst >> 32) & 0xFFFFFFFFL).toInt)
        axilite_driver_r.write(0x1c, (addr_src & 0xFFFFFFFFL).toInt)
        axilite_driver_r.write(0x20, ((addr_src >> 32) & 0xFFFFFFFFL).toInt)
        axilite_driver_r.write(0x28, (addr_smooth_scales & 0xFFFFFFFFL).toInt)
        axilite_driver_r.write(0x2c, ((addr_smooth_scales >> 32) & 0xFFFFFFFFL).toInt)
        
        // Write scalar parameters to s_axi_control
        // From synthesis report: s_axi_control has address width 5
        // ADDR_IN_DIM_DATA_0 = 5'h10
        // ADDR_NUM_PATCHES_DATA_0 = 5'h18
        println("Writing scalar parameters to s_axi_control...")
        axilite_driver.write(0x10, D_INNER) // in_dim (ADDR_IN_DIM_DATA_0)
        axilite_driver.write(0x18, NUM_PATCHES) // num_patches
        
        println("Configuration complete")
        dut.clockDomain.waitSampling(10)
        
        // Start computation
        println("Starting computation...")
        val ADDR_AP_CTRL = 0x00
        axilite_driver.write(ADDR_AP_CTRL, 0x11) // AP_START=1, AP_CONTINUE=1
        dut.clockDomain.waitSampling(10)
        axilite_driver.write(ADDR_AP_CTRL, 0x10) // Clear AP_START, keep AP_CONTINUE=1
        
        // Wait for completion
        println("Waiting for completion...")
        var done = false
        var cycles = 0
        val max_cycles = 1000000
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
        val dut_output_flat = new Array[Long](NUM_PATCHES * D_INNER)
        
        // Read data organized as fm_block_t blocks (each block is one 256-bit word)
        // Read each 32-bit element individually from its byte offset (little-endian)
        for (p <- 0 until NUM_PATCHES) {
          for (b <- 0 until NUM_FEATURE_BLOCKS_INNER) {
            val block_idx = p * NUM_FEATURE_BLOCKS_INNER + b
            val block_addr = (addr_dst + block_idx * 32).toLong
            // Read each 32-bit element individually from its byte offset (little-endian)
            for (o <- 0 until FEATURE_BLOCK_SIZE) {
              val global_idx = p * D_INNER + b * FEATURE_BLOCK_SIZE + o
              if (global_idx < dut_output_flat.length) {
                // Read 32-bit element from byte offset o*4 (little-endian byte order)
                val elem_addr = block_addr + o * 4
                val unsigned_value = axi_mem_out.memory.readBigInt(elem_addr, 4)
                // Convert from unsigned 32-bit to signed
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
        
        // Compare results
        println("Comparing results...")
        compare_arrays(ref_output, dut_output_flat, "SMOOTH Output", FixedPointTypes.fm_t)
        
        simSuccess()
        
        dut.clockDomain.waitSampling(100)
  }
}

