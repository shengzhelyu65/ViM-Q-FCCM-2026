import spinal.core.sim._
import spinal.core._
import spinal.lib.bus.amba4.axi.sim.{AxiMemorySim, AxiMemorySimConfig}
import spinal.lib.bus.amba4.axi._
import spinal.lib.bus.amba4.axilite.{AxiLite4, AxiLite4SpecRenamer}
import spinal.lib._
import utils._

import scala.language.postfixOps

class SSM_Blackbox extends BlackBox {
  val top_name: String = "SSM"
  setDefinitionName(top_name)
  
  private val verilog_file_path: String = s"src/main/verilog/$top_name/all.v"
  
  val io = new Bundle {
    val ap_clk: Bool = in Bool()
    val ap_rst_n: Bool = in Bool()
    
    val m_axi_in_u: BlackboxAxi = master(BlackboxAxi(BlackboxAxiConfig(verilog_file_path, "IN_U")))
    val m_axi_in_delta: BlackboxAxi = master(BlackboxAxi(BlackboxAxiConfig(verilog_file_path, "IN_DELTA")))
    val m_axi_in_z_silu: BlackboxAxi = master(BlackboxAxi(BlackboxAxiConfig(verilog_file_path, "IN_Z_SILU")))
    val m_axi_in_B: BlackboxAxi = master(BlackboxAxi(BlackboxAxiConfig(verilog_file_path, "IN_B")))
    val m_axi_in_C: BlackboxAxi = master(BlackboxAxi(BlackboxAxiConfig(verilog_file_path, "IN_C")))
    val m_axi_weights_A: BlackboxAxi = master(BlackboxAxi(BlackboxAxiConfig(verilog_file_path, "WEIGHTS_A")))
    val m_axi_weights_D: BlackboxAxi = master(BlackboxAxi(BlackboxAxiConfig(verilog_file_path, "WEIGHTS_D")))
    val m_axi_out_r: BlackboxAxi = master(BlackboxAxi(BlackboxAxiConfig(verilog_file_path, "OUT_R")))
    
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
  
  CustomBlackboxAxiRenamer(io.m_axi_in_u, "m_axi_in_u")
  CustomBlackboxAxiRenamer(io.m_axi_in_delta, "m_axi_in_delta")
  CustomBlackboxAxiRenamer(io.m_axi_in_z_silu, "m_axi_in_z_silu")
  CustomBlackboxAxiRenamer(io.m_axi_in_B, "m_axi_in_B")
  CustomBlackboxAxiRenamer(io.m_axi_in_C, "m_axi_in_C")
  CustomBlackboxAxiRenamer(io.m_axi_weights_A, "m_axi_weights_A")
  CustomBlackboxAxiRenamer(io.m_axi_weights_D, "m_axi_weights_D")
  CustomBlackboxAxiRenamer(io.m_axi_out_r, "m_axi_out_r")
  BlackboxAxiLiteRenamer(io.s_axi_control)
  
  io.interrupt := False
  
  mapClockDomain(clock = io.ap_clk, reset = io.ap_rst_n, resetActiveLevel = LOW)
  addRTLPath(verilog_file_path)
}

class SSM extends Component {
  val top_name: String = "SSM"
  setDefinitionName(top_name + "_wrapper")
  
  private val black_box = new SSM_Blackbox()
  
  val io = new Bundle {
    val signals: DaisyChain[ManagerSignals] = DaisyChain(ManagerSignals())
    
    val m_axi_in_u: Axi4 = master(Axi4(black_box.io.m_axi_in_u.config.to_std_config()))
    val m_axi_in_delta: Axi4 = master(Axi4(black_box.io.m_axi_in_delta.config.to_std_config()))
    val m_axi_in_z_silu: Axi4 = master(Axi4(black_box.io.m_axi_in_z_silu.config.to_std_config()))
    val m_axi_in_B: Axi4 = master(Axi4(black_box.io.m_axi_in_B.config.to_std_config()))
    val m_axi_in_C: Axi4 = master(Axi4(black_box.io.m_axi_in_C.config.to_std_config()))
    val m_axi_weights_A: Axi4 = master(Axi4(black_box.io.m_axi_weights_A.config.to_std_config()))
    val m_axi_weights_D: Axi4 = master(Axi4(black_box.io.m_axi_weights_D.config.to_std_config()))
    val m_axi_out_r: Axi4 = master(Axi4(black_box.io.m_axi_out_r.config.to_std_config()))
    
    val s_axi_control: AxiLite4 = slave(AxiLite4(black_box.io.s_axi_control.config.to_std_config()))
    val s_axi_control_r: AxiLite4 = slave(AxiLite4(addressWidth = 7, dataWidth = 32))
  }
  
  noIoPrefix()
  
  val manager = new Manager(single = true)
  manager.io.signals <> io.signals
  
  black_box.io.m_axi_in_u.connect2std(io.m_axi_in_u)
  black_box.io.m_axi_in_delta.connect2std(io.m_axi_in_delta)
  black_box.io.m_axi_in_z_silu.connect2std(io.m_axi_in_z_silu)
  black_box.io.m_axi_in_B.connect2std(io.m_axi_in_B)
  black_box.io.m_axi_in_C.connect2std(io.m_axi_in_C)
  black_box.io.m_axi_weights_A.connect2std(io.m_axi_weights_A)
  black_box.io.m_axi_weights_D.connect2std(io.m_axi_weights_D)
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
  
  Axi4SpecRenamer(io.m_axi_in_u)
  Axi4SpecRenamer(io.m_axi_in_delta)
  Axi4SpecRenamer(io.m_axi_in_z_silu)
  Axi4SpecRenamer(io.m_axi_in_B)
  Axi4SpecRenamer(io.m_axi_in_C)
  Axi4SpecRenamer(io.m_axi_weights_A)
  Axi4SpecRenamer(io.m_axi_weights_D)
  Axi4SpecRenamer(io.m_axi_out_r)
  AxiLite4SpecRenamer(io.s_axi_control)
  AxiLite4SpecRenamer(io.s_axi_control_r)
}

object simulate_ssm extends App {
  
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
    .compile(new SSM)
    .doSimUntilVoid { dut =>
      val D_INNER = 384
      val D_STATE = 16
      val SCAN_DIM = D_INNER * D_STATE
      val NUM_PATCHES = 197
      val LAYER_IDX = 0
      
      val data_path_prefix = utils.DataPathConfig.data_path_prefix
      
      val axi_mem_in_u = AxiMemorySim(dut.io.m_axi_in_u, dut.clockDomain, AxiMemorySimConfig())
      val axi_mem_in_delta = AxiMemorySim(dut.io.m_axi_in_delta, dut.clockDomain, AxiMemorySimConfig())
      val axi_mem_in_z_silu = AxiMemorySim(dut.io.m_axi_in_z_silu, dut.clockDomain, AxiMemorySimConfig())
      val axi_mem_in_B = AxiMemorySim(dut.io.m_axi_in_B, dut.clockDomain, AxiMemorySimConfig())
      val axi_mem_in_C = AxiMemorySim(dut.io.m_axi_in_C, dut.clockDomain, AxiMemorySimConfig())
      val axi_mem_weights_A = AxiMemorySim(dut.io.m_axi_weights_A, dut.clockDomain, AxiMemorySimConfig())
      val axi_mem_weights_D = AxiMemorySim(dut.io.m_axi_weights_D, dut.clockDomain, AxiMemorySimConfig())
      val axi_mem_out = AxiMemorySim(dut.io.m_axi_out_r, dut.clockDomain, AxiMemorySimConfig())
      
      val axilite_driver = AddressSeparableAxiLite4Driver(dut.io.s_axi_control, dut.clockDomain)
      val axilite_driver_r = AddressSeparableAxiLite4Driver(dut.io.s_axi_control_r, dut.clockDomain)
      
      val addr_u = 0x100000L
      val addr_delta = 0x200000L
      val addr_z_silu = 0x300000L
      val addr_B = 0x400000L
      val addr_C = 0x500000L
      val addr_A = 0x600000L
      val addr_D = 0x700000L
      val addr_dst = 0x800000L
      
      val ADDR_AP_CTRL = 0x00
      val ADDR_SCAN_DIM = 0x10
      val ADDR_INNER_DIM = 0x18
      val ADDR_NUM_PATCHES = 0x20
      
      init_clock(dut.clockDomain, 10)
      dut.clockDomain.waitSampling(10)
      
      println("=== SSM Simulation Test ===")
      
      println("Loading input data...")
      val u_file = s"${utils.DataPathConfig.data_path_prefix}/ssm_float32/layers.${LAYER_IDX}.mixer.scan.u.float32.bin"
      val delta_file = s"${utils.DataPathConfig.data_path_prefix}/ssm_float32/layers.${LAYER_IDX}.mixer.scan.delta.float32.bin"
      val z_silu_file = s"${utils.DataPathConfig.data_path_prefix}/ssm_float32/layers.${LAYER_IDX}.mixer.scan.z_silu.float32.bin"
      val B_file = s"${utils.DataPathConfig.data_path_prefix}/ssm_float32/layers.${LAYER_IDX}.mixer.scan.B.float32.bin"
      val C_file = s"${utils.DataPathConfig.data_path_prefix}/ssm_float32/layers.${LAYER_IDX}.mixer.scan.C.float32.bin"
      val A_file = s"${utils.DataPathConfig.data_path_prefix}/ssm_float32/layers.${LAYER_IDX}.mixer.scan.A.float32.bin"
      val D_file = s"${utils.DataPathConfig.data_path_prefix}/ssm_float32/layers.${LAYER_IDX}.mixer.scan.D.float32.bin"
      
      val u_float = read_float_file(u_file, NUM_PATCHES * D_INNER)
      val delta_float = read_float_file(delta_file, NUM_PATCHES * D_INNER)
      val z_silu_float = read_float_file(z_silu_file, NUM_PATCHES * D_INNER)
      val B_float = read_float_file(B_file, NUM_PATCHES * D_STATE)
      val C_float = read_float_file(C_file, NUM_PATCHES * D_STATE)
      val A_float = read_float_file(A_file, SCAN_DIM)
      val D_float = read_float_file(D_file, D_INNER)
      
      val u_data = u_float.map(f => FixedPointTypes.floatToFixed(f, FixedPointTypes.fm_t).toLong)
      val delta_data = delta_float.map(f => FixedPointTypes.floatToFixed(f, FixedPointTypes.fm_t).toLong)
      val z_silu_data = z_silu_float.map(f => FixedPointTypes.floatToFixed(f, FixedPointTypes.fm_t).toLong)
      val B_data = B_float.map(f => FixedPointTypes.floatToFixed(f, FixedPointTypes.fm_t).toLong)
      val C_data = C_float.map(f => FixedPointTypes.floatToFixed(f, FixedPointTypes.fm_t).toLong)
      val A_data = A_float.map(f => FixedPointTypes.floatToFixed(f, FixedPointTypes.scan_t).toLong)
      val D_data = D_float.map(f => FixedPointTypes.floatToFixed(f, FixedPointTypes.fm_t).toLong)
      
      println("Loading reference output...")
      val ref_file = s"${utils.DataPathConfig.data_path_prefix}/ssm_float32/layers.${LAYER_IDX}.mixer.scan.output.float32.bin"
      val ref_output_float = read_float_file(ref_file, NUM_PATCHES * D_INNER)
      val ref_output = ref_output_float.map(f => 
        FixedPointTypes.floatToFixed(f, FixedPointTypes.scan_t).toLong
      )
      
      println("Writing data to AXI memory...")
      val AXI_XFER_BIT_WIDTH_VAL = 256
      val FEATURE_BLOCK_SIZE = 8
      val NUM_FEATURE_BLOCKS_INNER = (D_INNER + FEATURE_BLOCK_SIZE - 1) / FEATURE_BLOCK_SIZE
      val NUM_FEATURE_BLOCKS_STATE = (D_STATE + FEATURE_BLOCK_SIZE - 1) / FEATURE_BLOCK_SIZE
      
      for (p <- 0 until NUM_PATCHES) {
        for (b <- 0 until NUM_FEATURE_BLOCKS_INNER) {
          val block_idx = p * NUM_FEATURE_BLOCKS_INNER + b
          val block_base_addr_u = (addr_u + block_idx * 32).toLong
          val block_base_addr_delta = (addr_delta + block_idx * 32).toLong
          val block_base_addr_z_silu = (addr_z_silu + block_idx * 32).toLong
          for (o <- 0 until FEATURE_BLOCK_SIZE) {
            val global_idx = p * D_INNER + b * FEATURE_BLOCK_SIZE + o
            if (global_idx < u_data.length) {
              val value_u = BigInt(u_data(global_idx))
              val value_delta = BigInt(delta_data(global_idx))
              val value_z_silu = BigInt(z_silu_data(global_idx))
              val unsigned_u = if (value_u < 0) (BigInt(1) << 32) + value_u else value_u
              val unsigned_delta = if (value_delta < 0) (BigInt(1) << 32) + value_delta else value_delta
              val unsigned_z_silu = if (value_z_silu < 0) (BigInt(1) << 32) + value_z_silu else value_z_silu
              val elem_addr_u = block_base_addr_u + o * 4
              val elem_addr_delta = block_base_addr_delta + o * 4
              val elem_addr_z_silu = block_base_addr_z_silu + o * 4
              axi_mem_in_u.memory.writeBigInt(elem_addr_u, unsigned_u, 4)
              axi_mem_in_delta.memory.writeBigInt(elem_addr_delta, unsigned_delta, 4)
              axi_mem_in_z_silu.memory.writeBigInt(elem_addr_z_silu, unsigned_z_silu, 4)
            }
          }
        }
        for (b <- 0 until NUM_FEATURE_BLOCKS_STATE) {
          val block_idx = p * NUM_FEATURE_BLOCKS_STATE + b
          val block_base_addr_B = (addr_B + block_idx * 32).toLong
          val block_base_addr_C = (addr_C + block_idx * 32).toLong
          for (o <- 0 until FEATURE_BLOCK_SIZE) {
            val global_idx = p * D_STATE + b * FEATURE_BLOCK_SIZE + o
            if (global_idx < B_data.length) {
              val value_B = BigInt(B_data(global_idx))
              val value_C = BigInt(C_data(global_idx))
              val unsigned_B = if (value_B < 0) (BigInt(1) << 32) + value_B else value_B
              val unsigned_C = if (value_C < 0) (BigInt(1) << 32) + value_C else value_C
              val elem_addr_B = block_base_addr_B + o * 4
              val elem_addr_C = block_base_addr_C + o * 4
              axi_mem_in_B.memory.writeBigInt(elem_addr_B, unsigned_B, 4)
              axi_mem_in_C.memory.writeBigInt(elem_addr_C, unsigned_C, 4)
            }
          }
        }
      }
      
      val A_ELEMS_PER_WORD = AXI_XFER_BIT_WIDTH_VAL / 32
      val A_words = (SCAN_DIM + A_ELEMS_PER_WORD - 1) / A_ELEMS_PER_WORD
      for (w <- 0 until A_words) {
        val word_base_addr = (addr_A + w * (AXI_XFER_BIT_WIDTH_VAL / 8)).toLong
        for (j <- 0 until A_ELEMS_PER_WORD) {
          val idx = w * A_ELEMS_PER_WORD + j
          if (idx < SCAN_DIM) {
            val value = BigInt(A_data(idx))
            val unsigned_value = if (value < 0) (BigInt(1) << 32) + value else value
            val elem_addr = word_base_addr + j * 4
            axi_mem_weights_A.memory.writeBigInt(elem_addr, unsigned_value, 4)
          }
        }
      }
      
      val D_ELEMS_PER_WORD = AXI_XFER_BIT_WIDTH_VAL / 32
      val D_words = (D_INNER + D_ELEMS_PER_WORD - 1) / D_ELEMS_PER_WORD
      for (w <- 0 until D_words) {
        val word_base_addr = (addr_D + w * (AXI_XFER_BIT_WIDTH_VAL / 8)).toLong
        for (j <- 0 until D_ELEMS_PER_WORD) {
          val idx = w * D_ELEMS_PER_WORD + j
          if (idx < D_INNER) {
            val value = BigInt(D_data(idx))
            val unsigned_value = if (value < 0) (BigInt(1) << 32) + value else value
            val elem_addr = word_base_addr + j * 4
            axi_mem_weights_D.memory.writeBigInt(elem_addr, unsigned_value, 4)
          }
        }
      }
      
      // Wait for all memory writes to complete before reset
      dut.clockDomain.waitSampling(10)
      
      // Reset AXI memory simulators
      axi_mem_in_u.reset()
      axi_mem_in_delta.reset()
      axi_mem_in_z_silu.reset()
      axi_mem_in_B.reset()
      axi_mem_in_C.reset()
      axi_mem_weights_A.reset()
      axi_mem_weights_D.reset()
      axi_mem_out.reset()
      axilite_driver.reset()
      axilite_driver_r.reset()
      
      // Wait longer after reset for AXI-Lite to be ready
      println("Waiting after reset for interfaces to stabilize...")
      dut.clockDomain.waitSampling(100)
      
      // --- Core Run Function ---
      def runIteration(iterName: String): Array[Long] = {
        println(s"\n--- Starting Iteration: $iterName ---")
        
        println("Configuring module via AXI-Lite...")
        axilite_driver_r.write(0x10, (addr_dst & 0xFFFFFFFFL).toInt)
        axilite_driver_r.write(0x14, ((addr_dst >> 32) & 0xFFFFFFFFL).toInt)
        axilite_driver_r.write(0x18, 1) // OUT_R_R_CTRL
        axilite_driver_r.write(0x1c, (addr_u & 0xFFFFFFFFL).toInt)
        axilite_driver_r.write(0x20, ((addr_u >> 32) & 0xFFFFFFFFL).toInt)
        axilite_driver_r.write(0x24, 1) // U_SRC_CTRL
        axilite_driver_r.write(0x28, (addr_delta & 0xFFFFFFFFL).toInt)
        axilite_driver_r.write(0x2c, ((addr_delta >> 32) & 0xFFFFFFFFL).toInt)
        axilite_driver_r.write(0x30, 1) // DELTA_SRC_CTRL
        axilite_driver_r.write(0x34, (addr_z_silu & 0xFFFFFFFFL).toInt)
        axilite_driver_r.write(0x38, ((addr_z_silu >> 32) & 0xFFFFFFFFL).toInt)
        axilite_driver_r.write(0x3c, 1) // Z_SILU_SRC_CTRL
        axilite_driver_r.write(0x40, (addr_A & 0xFFFFFFFFL).toInt)
        axilite_driver_r.write(0x44, ((addr_A >> 32) & 0xFFFFFFFFL).toInt)
        axilite_driver_r.write(0x48, 1) // A_BASE_CTRL
        axilite_driver_r.write(0x4c, (addr_B & 0xFFFFFFFFL).toInt)
        axilite_driver_r.write(0x50, ((addr_B >> 32) & 0xFFFFFFFFL).toInt)
        axilite_driver_r.write(0x54, 1) // B_SRC_CTRL
        axilite_driver_r.write(0x58, (addr_C & 0xFFFFFFFFL).toInt)
        axilite_driver_r.write(0x5c, ((addr_C >> 32) & 0xFFFFFFFFL).toInt)
        axilite_driver_r.write(0x60, 1) // C_SRC_CTRL
        axilite_driver_r.write(0x64, (addr_D & 0xFFFFFFFFL).toInt)
        axilite_driver_r.write(0x68, ((addr_D >> 32) & 0xFFFFFFFFL).toInt)
        axilite_driver_r.write(0x6c, 1) // D_BASE_CTRL
        
        axilite_driver.write(ADDR_SCAN_DIM, SCAN_DIM)
        axilite_driver.write(ADDR_INNER_DIM, D_INNER)
        axilite_driver.write(ADDR_NUM_PATCHES, NUM_PATCHES)
        
        dut.clockDomain.waitSampling(10)
        
        // Wait for IDLE
        var idle = false
        var idle_cycles = 0
        while (!idle && idle_cycles < 1000) {
          val ctrl = axilite_driver.read(ADDR_AP_CTRL)
          idle = (ctrl & 4) != 0
          if (!idle) {
            dut.clockDomain.waitSampling(10)
            idle_cycles += 10
          }
        }
        
        println("Starting computation...")
        axilite_driver.write(ADDR_AP_CTRL, 0x10) // AP_CONTINUE=1
        dut.clockDomain.waitSampling(5)
        axilite_driver.write(ADDR_AP_CTRL, 0x11) // AP_START=1, AP_CONTINUE=1
        dut.clockDomain.waitSampling(5)
        axilite_driver.write(ADDR_AP_CTRL, 0x10) // Clear AP_START
        
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
          println(s"ERROR: $iterName did not complete")
          simFailure()
        }
        
        println(s"Reading results for $iterName...")
        val out_data = new Array[Long](NUM_PATCHES * D_INNER)
        for (p <- 0 until NUM_PATCHES) {
          for (b <- 0 until NUM_FEATURE_BLOCKS_INNER) {
            val block_idx = p * NUM_FEATURE_BLOCKS_INNER + b
            val block_base_addr = (addr_dst + block_idx * 32).toLong
            for (o <- 0 until FEATURE_BLOCK_SIZE) {
              val global_idx = p * D_INNER + b * FEATURE_BLOCK_SIZE + o
              if (global_idx < out_data.length) {
                val elem_addr = block_base_addr + o * 4
                val unsigned_value = axi_mem_out.memory.readBigInt(elem_addr, 4)
                val signed_value = if (unsigned_value >= (BigInt(1) << 31)) unsigned_value - (BigInt(1) << 32) else unsigned_value
                out_data(global_idx) = signed_value.toLong
              }
            }
          }
        }
        out_data
      }

      // --- Execute Run ---
      val output = runIteration("SSM Computation")
      println("Comparing results for SSM Computation...")
      compare_arrays(ref_output, output, "SSM Output", FixedPointTypes.scan_t)

      simSuccess()
      dut.clockDomain.waitSampling(100)
    }
}
