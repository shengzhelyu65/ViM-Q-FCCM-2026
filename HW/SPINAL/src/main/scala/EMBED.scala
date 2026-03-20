import spinal.core.sim._
import spinal.core._
import spinal.lib.bus.amba4.axi.sim.{AxiMemorySim, AxiMemorySimConfig}
import spinal.lib.bus.amba4.axi._
import spinal.lib.bus.amba4.axilite.{AxiLite4, AxiLite4SpecRenamer}
import spinal.lib._
import utils._

import scala.language.postfixOps

class EMBED_Blackbox extends BlackBox {
  val top_name: String = "EMBED"
  setDefinitionName(top_name)
  
  private val verilog_file_path: String = s"src/main/verilog/$top_name/all.v"
  
  val io = new Bundle {
    val ap_clk: Bool = in Bool()
    val ap_rst_n: Bool = in Bool()
    
    val m_axi_out_r: BlackboxAxi = master(BlackboxAxi(BlackboxAxiConfig(verilog_file_path, "OUT_R")))
    val m_axi_in_r: BlackboxAxi = master(BlackboxAxi(BlackboxAxiConfig(verilog_file_path, "IN_R")))
    val m_axi_weights: BlackboxAxi = master(BlackboxAxi(BlackboxAxiConfig(verilog_file_path, "WEIGHTS")))
    
    val s_axi_control: BlackboxAxiLite = slave(BlackboxAxiLite(BlackboxAxiLiteConfig(verilog_file_path)))
    
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
    
    val interrupt: Bool = out Bool()
  }
  
  noIoPrefix()
  
  CustomBlackboxAxiRenamer(io.m_axi_out_r, "m_axi_out_r")
  CustomBlackboxAxiRenamer(io.m_axi_in_r, "m_axi_in_r")
  CustomBlackboxAxiRenamer(io.m_axi_weights, "m_axi_weights")
  BlackboxAxiLiteRenamer(io.s_axi_control)
  
  io.interrupt := False
  
  mapClockDomain(clock = io.ap_clk, reset = io.ap_rst_n, resetActiveLevel = LOW)
  addRTLPath(verilog_file_path)
}

class EMBED extends Component {
  val top_name: String = "EMBED"
  setDefinitionName(top_name + "_wrapper")
  
  private val black_box = new EMBED_Blackbox()
  
  val io = new Bundle {
    val signals: DaisyChain[ManagerSignals] = DaisyChain(ManagerSignals())
    
    val m_axi_out_r: Axi4 = master(Axi4(black_box.io.m_axi_out_r.config.to_std_config()))
    val m_axi_in_r: Axi4 = master(Axi4(black_box.io.m_axi_in_r.config.to_std_config()))
    val m_axi_weights: Axi4 = master(Axi4(black_box.io.m_axi_weights.config.to_std_config()))
    
    val s_axi_control: AxiLite4 = slave(AxiLite4(black_box.io.s_axi_control.config.to_std_config()))
    val s_axi_control_r: AxiLite4 = slave(AxiLite4(addressWidth = 7, dataWidth = 32))
  }
  
  noIoPrefix()
  
  val manager = new Manager(single = true)
  manager.io.signals <> io.signals
  
  black_box.io.m_axi_out_r.connect2std(io.m_axi_out_r)
  black_box.io.m_axi_in_r.connect2std(io.m_axi_in_r)
  black_box.io.m_axi_weights.connect2std(io.m_axi_weights)
  black_box.io.s_axi_control.connect2std(io.s_axi_control)
  
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
  
  Axi4SpecRenamer(io.m_axi_out_r)
  Axi4SpecRenamer(io.m_axi_in_r)
  Axi4SpecRenamer(io.m_axi_weights)
  AxiLite4SpecRenamer(io.s_axi_control)
  AxiLite4SpecRenamer(io.s_axi_control_r)
}

object simulate_embed extends App {
  
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
    .compile(new EMBED)
    .doSimUntilVoid { dut =>
      // Model parameters
      val D_MODEL = 192
      val NUM_PATCHES = 197
      val INPUT_HEIGHT = 224
      val INPUT_WIDTH = 224
      val INPUT_CHANNELS = 3
      val FEATURE_BLOCK_SIZE = 8
      
      // Data path
      val data_path_prefix = utils.DataPathConfig.data_path_prefix
      
      val axi_mem_out = AxiMemorySim(dut.io.m_axi_out_r, dut.clockDomain, AxiMemorySimConfig())
      val axi_mem_in = AxiMemorySim(dut.io.m_axi_in_r, dut.clockDomain, AxiMemorySimConfig())
      val axi_mem_weights = AxiMemorySim(dut.io.m_axi_weights, dut.clockDomain, AxiMemorySimConfig())
      
      val axilite_driver = AddressSeparableAxiLite4Driver(dut.io.s_axi_control, dut.clockDomain)
      val axilite_driver_r = AddressSeparableAxiLite4Driver(dut.io.s_axi_control_r, dut.clockDomain)
      
      // Memory addresses
      val addr_out = 0x100000L
      val addr_in = 0x200000L
      val addr_weights = 0x300000L
      val addr_bias = 0x400000L
      val addr_pos_embed = 0x500000L
      val addr_cls_token = 0x600000L
      
      // AXI-Lite register addresses
      val ADDR_AP_CTRL = 0x00
      val ADDR_DIM_DATA_0 = 0x10
      val ADDR_NUMPATCHES_DATA_0 = 0x18
      val ADDR_IMGHEIGHT_DATA_0 = 0x20
      val ADDR_IMGWIDTH_DATA_0 = 0x28
      
      // s_axi_control_r address offsets
      val ADDR_OUT_DATA_0 = 0x10
      val ADDR_IN_DATA_0 = 0x1c
      val ADDR_WEIGHTS_DATA_0 = 0x28
      val ADDR_BIAS_DATA_0 = 0x34
      val ADDR_POS_EMBED_DATA_0 = 0x40
      val ADDR_CLS_TOKEN_DATA_0 = 0x4c
      
      init_clock(dut.clockDomain, 10)
      dut.clockDomain.waitSampling(10)
      
      println("=== EMBED Simulation Test ===")
      println(s"D_MODEL: $D_MODEL, NUM_PATCHES: $NUM_PATCHES")
      println(s"INPUT: ${INPUT_HEIGHT}x${INPUT_WIDTH}x${INPUT_CHANNELS}")
      
      // Load image data
      println("Loading image data...")
      val image_file = s"$data_path_prefix/image_float32_block/image.float32.bin"
      val image_float = read_float_file(image_file, INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS)
      val image_data = image_float.map(f => 
        FixedPointTypes.floatToFixed(f, FixedPointTypes.pixel_t).toLong
      )
      
      // Load weights
      println("Loading weights...")
      val weight_file = s"$data_path_prefix/bin_float32_block/patch_embed.proj.weight.float32.bin"
      val weight_float = read_float_file(weight_file, D_MODEL * INPUT_CHANNELS * 16 * 16)
      val weight_data = weight_float.map(f => 
        FixedPointTypes.floatToFixed(f, FixedPointTypes.wt_patch_embed_t).toLong
      )
      
      // Load bias
      println("Loading bias...")
      val bias_file = s"$data_path_prefix/bin_float32_block/patch_embed.proj.bias.float32.bin"
      val bias_float = read_float_file(bias_file, D_MODEL)
      val bias_data = bias_float.map(f => 
        FixedPointTypes.floatToFixed(f, FixedPointTypes.wt_patch_bias_t).toLong
      )
      
      // Load pos_embed
      println("Loading pos_embed...")
      val pos_embed_file = s"$data_path_prefix/bin_float32_block/pos_embed.float32.bin"
      val pos_embed_float = read_float_file(pos_embed_file, NUM_PATCHES * D_MODEL)
      val pos_embed_data = pos_embed_float.map(f => 
        FixedPointTypes.floatToFixed(f, FixedPointTypes.fm_t).toLong
      )
      
      // Load cls_token
      println("Loading cls_token...")
      val cls_token_file = s"$data_path_prefix/bin_float32_block/cls_token.float32.bin"
      val cls_token_float = read_float_file(cls_token_file, D_MODEL)
      val cls_token_data = cls_token_float.map(f => 
        FixedPointTypes.floatToFixed(f, FixedPointTypes.fm_t).toLong
      )
      
      // Load reference output
      println("Loading reference output...")
      val ref_file = s"$data_path_prefix/ref_float32_block/layers.0.norm.input.float32.bin"
      val ref_output_float = read_float_file(ref_file, NUM_PATCHES * D_MODEL)
      val ref_output = ref_output_float.map(f => 
        FixedPointTypes.floatToFixed(f, FixedPointTypes.fm_t).toLong
      )
      
      // Write image data to AXI memory (pixel_t = 32-bit)
      println("Writing image data to AXI memory...")
      for (i <- image_data.indices) {
        val value = BigInt(image_data(i))
        val unsigned_value = if (value < 0) {
          (BigInt(1) << 32) + value
        } else {
          value
        }
        axi_mem_in.memory.writeBigInt(addr_in + i * 4, unsigned_value, 4)
      }
      
      // Write weights to AXI memory (wt_patch_embed_t = 16-bit)
      println("Writing weights to AXI memory...")
      for (i <- weight_data.indices) {
        val value = BigInt(weight_data(i))
        val unsigned_value = if (value < 0) {
          (BigInt(1) << 16) + value
        } else {
          value
        }
        axi_mem_weights.memory.writeBigInt(addr_weights + i * 2, unsigned_value, 2)
      }
      
      // Write bias to AXI memory (wt_patch_bias_t = 16-bit)
      println("Writing bias to AXI memory...")
      for (i <- bias_data.indices) {
        val value = BigInt(bias_data(i))
        val unsigned_value = if (value < 0) {
          (BigInt(1) << 16) + value
        } else {
          value
        }
        axi_mem_weights.memory.writeBigInt(addr_bias + i * 2, unsigned_value, 2)
      }
      
      // Write pos_embed to AXI memory (fm_t = 32-bit)
      println("Writing pos_embed to AXI memory...")
      for (i <- pos_embed_data.indices) {
        val value = BigInt(pos_embed_data(i))
        val unsigned_value = if (value < 0) {
          (BigInt(1) << 32) + value
        } else {
          value
        }
        axi_mem_weights.memory.writeBigInt(addr_pos_embed + i * 4, unsigned_value, 4)
      }
      
      // Write cls_token to AXI memory (fm_t = 32-bit)
      println("Writing cls_token to AXI memory...")
      for (i <- cls_token_data.indices) {
        val value = BigInt(cls_token_data(i))
        val unsigned_value = if (value < 0) {
          (BigInt(1) << 32) + value
        } else {
          value
        }
        axi_mem_weights.memory.writeBigInt(addr_cls_token + i * 4, unsigned_value, 4)
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
      axilite_driver_r.write(ADDR_OUT_DATA_0, (addr_out & 0xFFFFFFFFL).toInt)
      axilite_driver_r.write(ADDR_OUT_DATA_0 + 4, ((addr_out >> 32) & 0xFFFFFFFFL).toInt)
      
      axilite_driver_r.write(ADDR_IN_DATA_0, (addr_in & 0xFFFFFFFFL).toInt)
      axilite_driver_r.write(ADDR_IN_DATA_0 + 4, ((addr_in >> 32) & 0xFFFFFFFFL).toInt)
      
      axilite_driver_r.write(ADDR_WEIGHTS_DATA_0, (addr_weights & 0xFFFFFFFFL).toInt)
      axilite_driver_r.write(ADDR_WEIGHTS_DATA_0 + 4, ((addr_weights >> 32) & 0xFFFFFFFFL).toInt)
      
      axilite_driver_r.write(ADDR_BIAS_DATA_0, (addr_bias & 0xFFFFFFFFL).toInt)
      axilite_driver_r.write(ADDR_BIAS_DATA_0 + 4, ((addr_bias >> 32) & 0xFFFFFFFFL).toInt)
      
      axilite_driver_r.write(ADDR_POS_EMBED_DATA_0, (addr_pos_embed & 0xFFFFFFFFL).toInt)
      axilite_driver_r.write(ADDR_POS_EMBED_DATA_0 + 4, ((addr_pos_embed >> 32) & 0xFFFFFFFFL).toInt)
      
      axilite_driver_r.write(ADDR_CLS_TOKEN_DATA_0, (addr_cls_token & 0xFFFFFFFFL).toInt)
      axilite_driver_r.write(ADDR_CLS_TOKEN_DATA_0 + 4, ((addr_cls_token >> 32) & 0xFFFFFFFFL).toInt)
      
      // Write scalar parameters to s_axi_control
      println("Writing scalar parameters to s_axi_control...")
      axilite_driver.write(ADDR_DIM_DATA_0, D_MODEL)
      axilite_driver.write(ADDR_NUMPATCHES_DATA_0, NUM_PATCHES)
      axilite_driver.write(ADDR_IMGHEIGHT_DATA_0, INPUT_HEIGHT)
      axilite_driver.write(ADDR_IMGWIDTH_DATA_0, INPUT_WIDTH)
      
      println("Configuration complete")
      dut.clockDomain.waitSampling(10)
      
      // Start computation
      println("Starting computation...")
      axilite_driver.write(ADDR_AP_CTRL, 0x01)
      dut.clockDomain.waitSampling()
      
      // Wait for completion (poll AP_DONE)
      println("Waiting for completion...")
      var done = false
      var cycles = 0
      val max_cycles = 1000000
      while (!done && cycles < max_cycles) {
        val ctrl = axilite_driver.read(ADDR_AP_CTRL)
        done = (ctrl & 2) != 0
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
      val NUM_FEATURE_BLOCKS = (D_MODEL + FEATURE_BLOCK_SIZE - 1) / FEATURE_BLOCK_SIZE
      val total_blocks = NUM_PATCHES * NUM_FEATURE_BLOCKS
      val dut_output_flat = new Array[Long](NUM_PATCHES * D_MODEL)
      
      for (p <- 0 until NUM_PATCHES) {
        for (b <- 0 until NUM_FEATURE_BLOCKS) {
          val block_idx = p * NUM_FEATURE_BLOCKS + b
          val block_base_addr = (addr_out + block_idx * 32).toLong
          for (o <- 0 until FEATURE_BLOCK_SIZE) {
            val global_idx = p * D_MODEL + b * FEATURE_BLOCK_SIZE + o
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
      compare_arrays(ref_output, dut_output, "EMBED Output", FixedPointTypes.fm_t)
      
      simSuccess()
      
      dut.clockDomain.waitSampling(100)
    }
}
