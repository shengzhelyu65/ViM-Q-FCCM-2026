import spinal.core.sim._
import spinal.core._
import spinal.lib.bus.amba4.axi.sim.{AxiMemorySim, AxiMemorySimConfig}
import spinal.lib.bus.amba4.axi._
import spinal.lib.bus.amba4.axilite.{AxiLite4, AxiLite4SpecRenamer}
import spinal.lib._
import utils._

import scala.language.postfixOps

class PATCH_OPS_Blackbox extends BlackBox {
  val top_name: String = "PATCH_OPS"
  setDefinitionName(top_name)
  
  private val verilog_file_path: String = s"src/main/verilog/$top_name/all.v"
  
  val io = new Bundle {
    val ap_clk: Bool = in Bool()
    val ap_rst_n: Bool = in Bool()
    
    val m_axi_in_r: BlackboxAxi = master(BlackboxAxi(BlackboxAxiConfig(verilog_file_path, "IN_R")))
    val m_axi_out_r: BlackboxAxi = master(BlackboxAxi(BlackboxAxiConfig(verilog_file_path, "OUT_R")))
    
    val s_axi_control: BlackboxAxiLite = slave(BlackboxAxiLite(BlackboxAxiLiteConfig(verilog_file_path)))
    
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
    
    val interrupt: Bool = out Bool()
  }
  
  noIoPrefix()
  
  CustomBlackboxAxiRenamer(io.m_axi_in_r, "m_axi_in_r")
  CustomBlackboxAxiRenamer(io.m_axi_out_r, "m_axi_out_r")
  BlackboxAxiLiteRenamer(io.s_axi_control)
  
  io.interrupt := False
  
  mapClockDomain(clock = io.ap_clk, reset = io.ap_rst_n, resetActiveLevel = LOW)
  addRTLPath(verilog_file_path)
}

class PATCH_OPS extends Component {
  val top_name: String = "PATCH_OPS"
  setDefinitionName(top_name + "_wrapper")
  
  private val black_box = new PATCH_OPS_Blackbox()
  
  val io = new Bundle {
    val signals: DaisyChain[ManagerSignals] = DaisyChain(ManagerSignals())
    
    val m_axi_in_r: Axi4 = master(Axi4(black_box.io.m_axi_in_r.config.to_std_config()))
    val m_axi_out_r: Axi4 = master(Axi4(black_box.io.m_axi_out_r.config.to_std_config()))
    
    val s_axi_control: AxiLite4 = slave(AxiLite4(black_box.io.s_axi_control.config.to_std_config()))
    val s_axi_control_r: AxiLite4 = slave(AxiLite4(addressWidth = 6, dataWidth = 32))
  }
  
  noIoPrefix()
  
  val manager = new Manager(single = true)
  manager.io.signals <> io.signals
  
  black_box.io.m_axi_in_r.connect2std(io.m_axi_in_r)
  black_box.io.m_axi_out_r.connect2std(io.m_axi_out_r)
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
  
  AxiLite4SpecRenamer(io.s_axi_control_r)
}

object simulate_patch_ops extends App {
  
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
    .compile(new PATCH_OPS)
    .doSimUntilVoid { dut =>
      val D_MODEL = 192
      val D_INNER = 384
      val NUM_PATCHES = 197
      val LAYER_IDX = 0
      val CLS_TOKEN_IDX = 98
      
      // Data path
      val data_path_prefix = utils.DataPathConfig.data_path_prefix
      
      val axi_mem_in = AxiMemorySim(dut.io.m_axi_in_r, dut.clockDomain, AxiMemorySimConfig())
      val axi_mem_out = AxiMemorySim(dut.io.m_axi_out_r, dut.clockDomain, AxiMemorySimConfig())
      
      val axilite_driver = AddressSeparableAxiLite4Driver(dut.io.s_axi_control, dut.clockDomain)
      val axilite_driver_r = AddressSeparableAxiLite4Driver(dut.io.s_axi_control_r, dut.clockDomain)
      
      val addr_src = 0x100000L
      val addr_dst = 0x200000L
      
      val ADDR_AP_CTRL = 0x00
      val ADDR_MODE = 0x10
      val ADDR_NUM_PATCHES = 0x18
      val ADDR_CLS_TOKEN_IDX = 0x20
      val ADDR_INNER_DIM = 0x28
      val ADDR_MODEL_DIM = 0x30
      
      val PATCH_OP_FLIP = 0
      val PATCH_OP_LOAD_CLS = 1
      
      init_clock(dut.clockDomain, 10)
      dut.clockDomain.waitSampling(10)
      
      println("=== PATCH_OPS Simulation Test ===")
      
      def test_flip_patch(): Unit = {
        println("\n--- Testing FLIP_PATCH mode ---")
        
        val input_data = new Array[Long](NUM_PATCHES * D_INNER)
        for (p <- 0 until NUM_PATCHES) {
          for (d <- 0 until D_INNER) {
            val idx = p * D_INNER + d
            if (idx < input_data.length) {
              val value = FixedPointTypes.floatToFixed((p * 0.1f).toFloat, FixedPointTypes.fm_t)
              input_data(idx) = value.toLong
            }
          }
        }
        
        println("Generating reference output (flipped patches)...")
        val ref_output = new Array[Long](NUM_PATCHES * D_INNER)
        for (p <- 0 until NUM_PATCHES) {
          val flipped_patch = NUM_PATCHES - 1 - p
          for (d <- 0 until D_INNER) {
            val src_idx = flipped_patch * D_INNER + d
            val dst_idx = p * D_INNER + d
            if (src_idx < input_data.length && dst_idx < ref_output.length) {
              ref_output(dst_idx) = input_data(src_idx)
            }
          }
        }
        
        val FEATURE_BLOCK_SIZE = 8
        val NUM_FEATURE_BLOCKS_INNER = (D_INNER + FEATURE_BLOCK_SIZE - 1) / FEATURE_BLOCK_SIZE
        
        println("Writing input data to AXI memory...")
        for (p <- 0 until NUM_PATCHES) {
          for (b <- 0 until NUM_FEATURE_BLOCKS_INNER) {
            val block_idx = p * NUM_FEATURE_BLOCKS_INNER + b
            val block_base_addr = (addr_src + block_idx * 32).toLong
            for (o <- 0 until FEATURE_BLOCK_SIZE) {
              val global_idx = p * D_INNER + b * FEATURE_BLOCK_SIZE + o
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
        
        axi_mem_in.reset()
        axi_mem_out.reset()
        axilite_driver.reset()
        axilite_driver_r.reset()
        
        dut.clockDomain.waitSampling(100)
        
        println("Configuring module...")
        axilite_driver_r.write(0x10, (addr_dst & 0xFFFFFFFFL).toInt)
        axilite_driver_r.write(0x14, ((addr_dst >> 32) & 0xFFFFFFFFL).toInt)
        axilite_driver_r.write(0x1c, (addr_src & 0xFFFFFFFFL).toInt)
        axilite_driver_r.write(0x20, ((addr_src >> 32) & 0xFFFFFFFFL).toInt)
        
        axilite_driver.write(ADDR_MODE, PATCH_OP_FLIP)
        axilite_driver.write(ADDR_NUM_PATCHES, NUM_PATCHES)
        axilite_driver.write(ADDR_CLS_TOKEN_IDX, CLS_TOKEN_IDX)
        axilite_driver.write(ADDR_INNER_DIM, D_INNER)
        axilite_driver.write(ADDR_MODEL_DIM, D_MODEL)
        
        println("Starting computation...")
        axilite_driver.write(ADDR_AP_CTRL, 0x11)
        dut.clockDomain.waitSampling()
        axilite_driver.write(ADDR_AP_CTRL, 0x10)
        
        var done = false
        var cycles = 0
        val max_cycles = 1000000
        while (!done && cycles < max_cycles) {
          val ctrl = axilite_driver.read(ADDR_AP_CTRL)
          done = (ctrl & 2) != 0
          if (cycles == 0 || cycles == 1000 || (cycles % 10000 == 0 && cycles > 1000)) {
            println(s"Cycle $cycles: AP_CTRL=0x${ctrl.toString(16)}")
          }
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
        
        println("Reading results...")
        val dut_output_flat = new Array[Long](NUM_PATCHES * D_INNER)
        
        for (p <- 0 until NUM_PATCHES) {
          for (b <- 0 until NUM_FEATURE_BLOCKS_INNER) {
            val block_idx = p * NUM_FEATURE_BLOCKS_INNER + b
            val block_base_addr = (addr_dst + block_idx * 32).toLong
            for (o <- 0 until FEATURE_BLOCK_SIZE) {
              val global_idx = p * D_INNER + b * FEATURE_BLOCK_SIZE + o
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
        
        println("Comparing results...")
        compare_arrays(ref_output, dut_output_flat, "FLIP_PATCH Output", FixedPointTypes.fm_t)
      }
      
      def test_load_cls_token(): Unit = {
        println("\n--- Testing LOAD_CLS_TOKEN mode ---")
        
        println("Loading input data...")
        val input_file = s"$data_path_prefix/ref_float32_block/layers.${LAYER_IDX}.norm.output.float32.bin"
        val input_float = read_float_file(input_file, NUM_PATCHES * D_MODEL)
        val input_data = input_float.map(f => 
          FixedPointTypes.floatToFixed(f, FixedPointTypes.fm_t).toLong
        )
        
        val FEATURE_BLOCK_SIZE = 8
        val NUM_FEATURE_BLOCKS = (D_MODEL + FEATURE_BLOCK_SIZE - 1) / FEATURE_BLOCK_SIZE
        
        println("Writing input data to AXI memory...")
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
        
        val ref_cls_token = new Array[Long](D_MODEL)
        for (d <- 0 until D_MODEL) {
          val src_idx = CLS_TOKEN_IDX * D_MODEL + d
          if (src_idx < input_data.length) {
            ref_cls_token(d) = input_data(src_idx)
          }
        }
        
        axi_mem_in.reset()
        axi_mem_out.reset()
        axilite_driver.reset()
        axilite_driver_r.reset()
        
        dut.clockDomain.waitSampling(100)
        
        println("Configuring module...")
        axilite_driver_r.write(0x10, (addr_dst & 0xFFFFFFFFL).toInt)
        axilite_driver_r.write(0x14, ((addr_dst >> 32) & 0xFFFFFFFFL).toInt)
        axilite_driver_r.write(0x1c, (addr_src & 0xFFFFFFFFL).toInt)
        axilite_driver_r.write(0x20, ((addr_src >> 32) & 0xFFFFFFFFL).toInt)
        
        axilite_driver.write(ADDR_MODE, PATCH_OP_LOAD_CLS)
        axilite_driver.write(ADDR_NUM_PATCHES, NUM_PATCHES)
        axilite_driver.write(ADDR_CLS_TOKEN_IDX, CLS_TOKEN_IDX)
        axilite_driver.write(ADDR_INNER_DIM, D_INNER)
        axilite_driver.write(ADDR_MODEL_DIM, D_MODEL)
        
        println("Starting computation...")
        axilite_driver.write(ADDR_AP_CTRL, 0x11)
        dut.clockDomain.waitSampling()
        axilite_driver.write(ADDR_AP_CTRL, 0x10)
        
        var done = false
        var cycles = 0
        val max_cycles = 1000000
        while (!done && cycles < max_cycles) {
          val ctrl = axilite_driver.read(ADDR_AP_CTRL)
          done = (ctrl & 2) != 0
          if (cycles == 0 || cycles == 1000 || (cycles % 10000 == 0 && cycles > 1000)) {
            println(s"Cycle $cycles: AP_CTRL=0x${ctrl.toString(16)}")
          }
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
        
        println("Reading results...")
        val dut_output_flat = new Array[Long](D_MODEL)
        
        for (b <- 0 until NUM_FEATURE_BLOCKS) {
          val block_base_addr = (addr_dst + b * 32).toLong
          for (o <- 0 until FEATURE_BLOCK_SIZE) {
            val global_idx = b * FEATURE_BLOCK_SIZE + o
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
        
        println("Comparing results...")
        compare_arrays(ref_cls_token, dut_output_flat, "LOAD_CLS_TOKEN Output", FixedPointTypes.fm_t)
      }
      
      test_flip_patch()
      test_load_cls_token()
      
      println("\n=== PATCH_OPS Simulation Test Completed ===")
      simSuccess()
      
      dut.clockDomain.waitSampling(100)
    }
}

