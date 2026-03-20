import spinal.core.sim._
import spinal.core._
import spinal.lib.bus.amba4.axi.sim.{AxiMemorySim, AxiMemorySimConfig}
import spinal.lib.bus.amba4.axi._
import spinal.lib.bus.amba4.axilite.{AxiLite4, AxiLite4SpecRenamer}
import spinal.lib._
import utils._

import scala.language.postfixOps

class NORM_SUM_Blackbox extends BlackBox {
  val top_name: String = "NORM_SUM"
  setDefinitionName(top_name)
  
  private val verilog_file_path: String = s"src/main/verilog/$top_name/all.v"
  
  val io = new Bundle {
    val ap_clk: Bool = in Bool()
    val ap_rst_n: Bool = in Bool()
    
    val m_axi_in_a: BlackboxAxi = master(BlackboxAxi(BlackboxAxiConfig(verilog_file_path, "IN_A")))
    val m_axi_in_b: BlackboxAxi = master(BlackboxAxi(BlackboxAxiConfig(verilog_file_path, "IN_B")))
    val m_axi_out_r: BlackboxAxi = master(BlackboxAxi(BlackboxAxiConfig(verilog_file_path, "OUT_R")))
    val m_axi_weights: BlackboxAxi = master(BlackboxAxi(BlackboxAxiConfig(verilog_file_path, "WEIGHTS")))
    
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
  
  CustomBlackboxAxiRenamer(io.m_axi_in_a, "m_axi_in_a")
  CustomBlackboxAxiRenamer(io.m_axi_in_b, "m_axi_in_b")
  CustomBlackboxAxiRenamer(io.m_axi_out_r, "m_axi_out_r")
  CustomBlackboxAxiRenamer(io.m_axi_weights, "m_axi_weights")
  BlackboxAxiLiteRenamer(io.s_axi_control)
  
  io.interrupt := False
  
  mapClockDomain(clock = io.ap_clk, reset = io.ap_rst_n, resetActiveLevel = LOW)
  addRTLPath(verilog_file_path)
}

class NORM_SUM extends Component {
  val top_name: String = "NORM_SUM"
  setDefinitionName(top_name + "_wrapper")
  
  private val black_box = new NORM_SUM_Blackbox()
  
  val io = new Bundle {
    val signals: DaisyChain[ManagerSignals] = DaisyChain(ManagerSignals())
    
    val m_axi_in_a: Axi4 = master(Axi4(black_box.io.m_axi_in_a.config.to_std_config()))
    val m_axi_in_b: Axi4 = master(Axi4(black_box.io.m_axi_in_b.config.to_std_config()))
    val m_axi_out_r: Axi4 = master(Axi4(black_box.io.m_axi_out_r.config.to_std_config()))
    val m_axi_weights: Axi4 = master(Axi4(black_box.io.m_axi_weights.config.to_std_config()))
    
    val s_axi_control: AxiLite4 = slave(AxiLite4(black_box.io.s_axi_control.config.to_std_config()))
    val s_axi_control_r: AxiLite4 = slave(AxiLite4(addressWidth = 6, dataWidth = 32))
  }
  
  noIoPrefix()
  
  val manager = new Manager(single = true)
  manager.io.signals <> io.signals
  
  black_box.io.m_axi_in_a.connect2std(io.m_axi_in_a)
  black_box.io.m_axi_in_b.connect2std(io.m_axi_in_b)
  black_box.io.m_axi_out_r.connect2std(io.m_axi_out_r)
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
  
  Axi4SpecRenamer(io.m_axi_in_a)
  Axi4SpecRenamer(io.m_axi_in_b)
  Axi4SpecRenamer(io.m_axi_out_r)
  Axi4SpecRenamer(io.m_axi_weights)
  AxiLite4SpecRenamer(io.s_axi_control)
  AxiLite4SpecRenamer(io.s_axi_control_r)
}

object simulate_norm_sum extends App {
  
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
    .addSimulatorFlag("-O3 --x-assign fast --x-initial fast --noassert --no-timing -Wno-fatal")
    .compile(new NORM_SUM)
    .doSimUntilVoid { dut =>
      val D_MODEL = 192
      val NUM_PATCHES = 197
      
      val axi_mem_in_a = AxiMemorySim(dut.io.m_axi_in_a, dut.clockDomain, AxiMemorySimConfig())
      val axi_mem_in_b = AxiMemorySim(dut.io.m_axi_in_b, dut.clockDomain, AxiMemorySimConfig())
      val axi_mem_out = AxiMemorySim(dut.io.m_axi_out_r, dut.clockDomain, AxiMemorySimConfig())
      val axi_mem_weights = AxiMemorySim(dut.io.m_axi_weights, dut.clockDomain, AxiMemorySimConfig())
      
      val axilite_driver = AddressSeparableAxiLite4Driver(dut.io.s_axi_control, dut.clockDomain)
      val axilite_driver_r = AddressSeparableAxiLite4Driver(dut.io.s_axi_control_r, dut.clockDomain)
      
      val addr_src_a = 0x10000000L
      val addr_src_b = 0x20000000L
      val addr_dst = 0x30000000L
      val addr_weights = 0x40000000L
      
      val ADDR_AP_CTRL = 0x00
      val ADDR_MODEL_DIM = 0x10
      val ADDR_NUM_PATCHES = 0x18
      val ADDR_MODE = 0x20
      
      val ADDR_DST_PTR = 0x10
      val ADDR_SRC_A_PTR = 0x1c
      val ADDR_SRC_B_PTR = 0x28
      val ADDR_WEIGHTS_PTR = 0x34
      
      init_clock(dut.clockDomain, 10)
      dut.clockDomain.waitSampling(10)
      
      println("=== NORM_SUM Simulation Test ===")
      
      val modes = Array(0, 1, 2, 3)
      val mode_names = Array("BOTH", "NORM_ONLY", "ADD_ONLY", "DIV2_ONLY")
      
      for (m <- modes.indices) {
        val test_mode = modes(m)
        println(s"\n--- Testing Mode: ${mode_names(m)} ---")
        
        // Generate test data
        val input_a_float = Array.tabulate(NUM_PATCHES * D_MODEL)(i => (i % 100) * 0.01f)
        val input_b_float = Array.tabulate(NUM_PATCHES * D_MODEL)(i => ((i + 50) % 100) * 0.01f)
        val weights_float = Array.fill(D_MODEL)(0.5f)
        
        val input_a = input_a_float.map(f => FixedPointTypes.floatToFixed(f, FixedPointTypes.fm_t).toLong)
        val input_b = input_b_float.map(f => FixedPointTypes.floatToFixed(f, FixedPointTypes.fm_t).toLong)
        val weights = weights_float.map(f => FixedPointTypes.floatToFixed(f, FixedPointTypes.wt_norm_t).toLong)
        
        // Reference calculation
        val ref_output = new Array[Long](NUM_PATCHES * D_MODEL)
        for (p <- 0 until NUM_PATCHES) {
          var sum_sq = 1e-5
          val patch_vals = new Array[Double](D_MODEL)
          for (i <- 0 until D_MODEL) {
            val idx = p * D_MODEL + i
            val val_a = input_a_float(idx)
            val val_b = input_b_float(idx)
            val val_sum = if (test_mode == 1) val_a else val_a + val_b
            patch_vals(i) = val_sum
            sum_sq += val_sum * val_sum / D_MODEL
          }
          val rms_inv = 1.0 / scala.math.sqrt(sum_sq)
          for (i <- 0 until D_MODEL) {
            val idx = p * D_MODEL + i
            val val_sum = patch_vals(i)
            val ref_f = test_mode match {
              case 0 => val_sum * rms_inv * weights_float(i)
              case 1 => val_sum * rms_inv * weights_float(i)
              case 2 => val_sum
              case 3 => val_sum * 0.5
            }
            ref_output(idx) = FixedPointTypes.floatToFixed(ref_f.toFloat, FixedPointTypes.fm_t).toLong
          }
        }
        
        // Write data to memory
        val FEATURE_BLOCK_SIZE = 8
        val NUM_FEATURE_BLOCKS = (D_MODEL + FEATURE_BLOCK_SIZE - 1) / FEATURE_BLOCK_SIZE
        
        for (p <- 0 until NUM_PATCHES) {
          for (b <- 0 until NUM_FEATURE_BLOCKS) {
            val block_idx = p * NUM_FEATURE_BLOCKS + b
            val base_a = addr_src_a + block_idx * 32
            val base_b = addr_src_b + block_idx * 32
            for (i <- 0 until FEATURE_BLOCK_SIZE) {
              val idx = p * D_MODEL + b * FEATURE_BLOCK_SIZE + i
              if (idx < input_a.length) {
                axi_mem_in_a.memory.writeBigInt(base_a + i * 4, BigInt(input_a(idx)) & 0xFFFFFFFFL, 4)
                axi_mem_in_b.memory.writeBigInt(base_b + i * 4, BigInt(input_b(idx)) & 0xFFFFFFFFL, 4)
              }
            }
          }
        }
        
        val WEIGHTS_PER_WORD = 256 / 16
        val weight_words = (D_MODEL + WEIGHTS_PER_WORD - 1) / WEIGHTS_PER_WORD
        for (w <- 0 until weight_words) {
          val base = addr_weights + w * 32
          for (j <- 0 until WEIGHTS_PER_WORD) {
            val idx = w * WEIGHTS_PER_WORD + j
            if (idx < D_MODEL) {
              axi_mem_weights.memory.writeBigInt(base + j * 2, BigInt(weights(idx)) & 0xFFFFL, 2)
            }
          }
        }
        
        axi_mem_in_a.reset()
        axi_mem_in_b.reset()
        axi_mem_out.reset()
        axi_mem_weights.reset()
        axilite_driver.reset()
        axilite_driver_r.reset()
        
        dut.clockDomain.waitSampling(100)
        
        // Configure
        axilite_driver_r.write(ADDR_DST_PTR, (addr_dst & 0xFFFFFFFFL).toInt)
        axilite_driver_r.write(ADDR_DST_PTR + 4, ((addr_dst >> 32) & 0xFFFFFFFFL).toInt)
        axilite_driver_r.write(ADDR_SRC_A_PTR, (addr_src_a & 0xFFFFFFFFL).toInt)
        axilite_driver_r.write(ADDR_SRC_A_PTR + 4, ((addr_src_a >> 32) & 0xFFFFFFFFL).toInt)
        axilite_driver_r.write(ADDR_SRC_B_PTR, (addr_src_b & 0xFFFFFFFFL).toInt)
        axilite_driver_r.write(ADDR_SRC_B_PTR + 4, ((addr_src_b >> 32) & 0xFFFFFFFFL).toInt)
        axilite_driver_r.write(ADDR_WEIGHTS_PTR, (addr_weights & 0xFFFFFFFFL).toInt)
        axilite_driver_r.write(ADDR_WEIGHTS_PTR + 4, ((addr_weights >> 32) & 0xFFFFFFFFL).toInt)
        
        axilite_driver.write(ADDR_MODEL_DIM, D_MODEL)
        axilite_driver.write(ADDR_NUM_PATCHES, NUM_PATCHES)
        axilite_driver.write(ADDR_MODE, test_mode)
        
        // Start
        axilite_driver.write(ADDR_AP_CTRL, 0x11)
        dut.clockDomain.waitSampling()
        axilite_driver.write(ADDR_AP_CTRL, 0x10)
        
        // Wait
        var done = false
        var cycles = 0
        while (!done && cycles < 1000000) {
          val ctrl = axilite_driver.read(ADDR_AP_CTRL)
          done = (ctrl & 2) != 0
          dut.clockDomain.waitSampling(100)
          cycles += 100
        }
        
        if (!done) {
          println(s"ERROR: Mode ${mode_names(m)} timed out!")
          simFailure()
        }
        println(s"Mode ${mode_names(m)} completed in $cycles cycles")
        
        // Read and compare
        val dut_output = new Array[Long](NUM_PATCHES * D_MODEL)
        for (p <- 0 until NUM_PATCHES) {
          for (b <- 0 until NUM_FEATURE_BLOCKS) {
            val base = addr_dst + (p * NUM_FEATURE_BLOCKS + b) * 32
            for (i <- 0 until FEATURE_BLOCK_SIZE) {
              val idx = p * D_MODEL + b * FEATURE_BLOCK_SIZE + i
              if (idx < dut_output.length) {
                val unsigned = axi_mem_out.memory.readBigInt(base + i * 4, 4)
                dut_output(idx) = if (unsigned >= (BigInt(1) << 31)) (unsigned - (BigInt(1) << 32)).toLong else unsigned.toLong
              }
            }
          }
        }
        
        compare_arrays(ref_output, dut_output, mode_names(m), FixedPointTypes.fm_t)
      }
      
      println("\n=== NORM_SUM Simulation Test Completed Successfully ===")
      simSuccess()
    }
}
