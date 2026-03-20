import spinal.core.sim._
import spinal.core._
import spinal.lib.bus.amba4.axi.sim.{AxiMemorySim, AxiMemorySimConfig}
import spinal.lib.bus.amba4.axi._
import spinal.lib.bus.amba4.axilite.{AxiLite4, AxiLite4SpecRenamer}
import spinal.lib._
import utils._

import scala.language.postfixOps

class CONV_Blackbox extends BlackBox {
  val top_name: String = "CONV"
  setDefinitionName(top_name)
  
  private val verilog_file_path: String = s"src/main/verilog/$top_name/all.v"
  
  val io = new Bundle {
    val ap_clk: Bool = in Bool()
    val ap_rst_n: Bool = in Bool()
    
    val m_axi_in_r: BlackboxAxi = master(BlackboxAxi(BlackboxAxiConfig(verilog_file_path, "IN_R")))
    val m_axi_out_r: BlackboxAxi = master(BlackboxAxi(BlackboxAxiConfig(verilog_file_path, "OUT_R")))
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
  
  CustomBlackboxAxiRenamer(io.m_axi_in_r, "m_axi_in_r")
  CustomBlackboxAxiRenamer(io.m_axi_out_r, "m_axi_out_r")
  CustomBlackboxAxiRenamer(io.m_axi_weights, "m_axi_weights")
  BlackboxAxiLiteRenamer(io.s_axi_control)
  
  io.interrupt := False
  
  mapClockDomain(clock = io.ap_clk, reset = io.ap_rst_n, resetActiveLevel = LOW)
  addRTLPath(verilog_file_path)
}

class CONV extends Component {
  val top_name: String = "CONV"
  setDefinitionName(top_name + "_wrapper")
  
  private val black_box = new CONV_Blackbox()
  
  val io = new Bundle {
    val signals: DaisyChain[ManagerSignals] = DaisyChain(ManagerSignals())
    
    val m_axi_in_r: Axi4 = master(Axi4(black_box.io.m_axi_in_r.config.to_std_config()))
    val m_axi_out_r: Axi4 = master(Axi4(black_box.io.m_axi_out_r.config.to_std_config()))
    val m_axi_weights: Axi4 = master(Axi4(black_box.io.m_axi_weights.config.to_std_config()))
    
    val s_axi_control: AxiLite4 = slave(AxiLite4(black_box.io.s_axi_control.config.to_std_config()))
    val s_axi_control_r: AxiLite4 = slave(AxiLite4(addressWidth = 7, dataWidth = 32))
  }
  
  noIoPrefix()
  
  val manager = new Manager(single = true)
  manager.io.signals <> io.signals
  
  black_box.io.m_axi_in_r.connect2std(io.m_axi_in_r)
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
  
  Axi4SpecRenamer(io.m_axi_in_r)
  Axi4SpecRenamer(io.m_axi_out_r)
  Axi4SpecRenamer(io.m_axi_weights)
  AxiLite4SpecRenamer(io.s_axi_control)
  AxiLite4SpecRenamer(io.s_axi_control_r)
}

object simulate_conv extends App {
  
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
    .compile(new CONV)
    .doSimUntilVoid { dut =>
      val D_INNER = 384
      val NUM_PATCHES = 197
      val CONV_KERNEL_SIZE = 4
      val LAYER_IDX = 0
      
      // Data path
      val data_path_prefix = utils.DataPathConfig.data_path_prefix
      
      val axi_mem_in = AxiMemorySim(dut.io.m_axi_in_r, dut.clockDomain, AxiMemorySimConfig())
      val axi_mem_out = AxiMemorySim(dut.io.m_axi_out_r, dut.clockDomain, AxiMemorySimConfig())
      val axi_mem_weights = AxiMemorySim(dut.io.m_axi_weights, dut.clockDomain, AxiMemorySimConfig())
      
      // Debug: Track AXI memory access (use @volatile for thread safety)
      @volatile var read_req_count_in = 0
      @volatile var read_resp_count_in = 0
      @volatile var write_req_count_out = 0
      @volatile var write_resp_count_out = 0
      @volatile var read_req_count_weights = 0
      @volatile var read_resp_count_weights = 0
      
      fork {
        while (true) {
          dut.clockDomain.waitSampling()
          // Track read requests on in_r
          if (dut.io.m_axi_in_r.ar.valid.toBoolean && dut.io.m_axi_in_r.ar.ready.toBoolean) {
            read_req_count_in += 1
            val addr = dut.io.m_axi_in_r.ar.payload.addr.toBigInt
            val len = dut.io.m_axi_in_r.ar.payload.len.toInt
            val size = dut.io.m_axi_in_r.ar.payload.size.toInt
            println(f"[AXI_IN_R] Read request #$read_req_count_in: addr=0x$addr%x, len=$len, size=$size")
          }
          // Track read responses on in_r
          if (dut.io.m_axi_in_r.r.valid.toBoolean && dut.io.m_axi_in_r.r.ready.toBoolean) {
            read_resp_count_in += 1
            if (read_resp_count_in % 100 == 0 || read_resp_count_in <= 10) {
              println(f"[AXI_IN_R] Read response #$read_resp_count_in")
            }
          }
          // Track write requests on out_r
          if (dut.io.m_axi_out_r.aw.valid.toBoolean && dut.io.m_axi_out_r.aw.ready.toBoolean) {
            write_req_count_out += 1
            val addr = dut.io.m_axi_out_r.aw.payload.addr.toBigInt
            val len = dut.io.m_axi_out_r.aw.payload.len.toInt
            println(f"[AXI_OUT_R] Write request #$write_req_count_out: addr=0x$addr%x, len=$len")
          }
          // Track write responses on out_r
          if (dut.io.m_axi_out_r.b.valid.toBoolean && dut.io.m_axi_out_r.b.ready.toBoolean) {
            write_resp_count_out += 1
            if (write_resp_count_out % 100 == 0 || write_resp_count_out <= 10) {
              println(f"[AXI_OUT_R] Write response #$write_resp_count_out")
            }
          }
          // Track read requests on weights
          if (dut.io.m_axi_weights.ar.valid.toBoolean && dut.io.m_axi_weights.ar.ready.toBoolean) {
            read_req_count_weights += 1
            val addr = dut.io.m_axi_weights.ar.payload.addr.toBigInt
            val len = dut.io.m_axi_weights.ar.payload.len.toInt
            val expected_beats = len + 1
            println(f"[AXI_WEIGHTS] Read request #$read_req_count_weights: addr=0x$addr%x, len=$len (expecting $expected_beats beats)")
          }
          // Track read responses on weights
          if (dut.io.m_axi_weights.r.valid.toBoolean && dut.io.m_axi_weights.r.ready.toBoolean) {
            read_resp_count_weights += 1
            val last = dut.io.m_axi_weights.r.payload.last.toBoolean
            if (read_resp_count_weights <= 20 || read_resp_count_weights % 50 == 0 || last) {
              println(f"[AXI_WEIGHTS] Read response #$read_resp_count_weights (LAST=$last)")
            }
          }
        }
      }
      
      val axilite_driver = AddressSeparableAxiLite4Driver(dut.io.s_axi_control, dut.clockDomain)
      val axilite_driver_r = AddressSeparableAxiLite4Driver(dut.io.s_axi_control_r, dut.clockDomain)
      
      val addr_src = 0x100000L
      val addr_dst = 0x200000L
      val addr_weights_mags = 0x300000L
      val addr_weights_signs = 0x400000L
      val addr_bias = 0x500000L
      val addr_scales = 0x600000L
      
      val ADDR_AP_CTRL = 0x00
      val ADDR_CONV_DIM = 0x10
      val ADDR_NUM_PATCHES = 0x18
      
      init_clock(dut.clockDomain, 10)
      dut.clockDomain.waitSampling(10)
      
      println("=== CONV Simulation Test ===")
      
      println("Loading input data...")
      val input_file = s"${utils.DataPathConfig.data_path_prefix}/ref_float32_block/layers.${LAYER_IDX}.mixer.in_proj.output.x.float32.bin"
      val mag_file = s"${utils.DataPathConfig.data_path_prefix}/bin_float32_block/layers.${LAYER_IDX}.mixer.conv1d.weight.magnitude.bin"
      val sign_file = s"${utils.DataPathConfig.data_path_prefix}/bin_float32_block/layers.${LAYER_IDX}.mixer.conv1d.weight.sign.bin"
      val bias_file = s"${utils.DataPathConfig.data_path_prefix}/bin_float32_block/layers.${LAYER_IDX}.mixer.conv1d.bias.float32.bin"
      val scale_file = s"${utils.DataPathConfig.data_path_prefix}/bin_float32_block/layers.${LAYER_IDX}.mixer.conv1d.weight_scale.float32.bin"
      
      val total_weight_elems = CONV_KERNEL_SIZE * D_INNER
      val weights_mags_int4 = read_int4_file(mag_file, total_weight_elems)
      val weights_signs_bit = read_bit_file(sign_file, total_weight_elems)
      val bias_float = read_float_file(bias_file, D_INNER)
      val scale_float = read_float_file(scale_file, D_INNER)
      
      // Convert input data from float to fixed point
      val input_float = read_float_file(input_file, NUM_PATCHES * D_INNER)
      val input_data = input_float.map(f => 
        FixedPointTypes.floatToFixed(f, FixedPointTypes.fm_t).toLong
      )
      
      // Convert bias and scale from float to fixed point
      val bias_fixed = bias_float.map(f => 
        FixedPointTypes.floatToFixed(f, FixedPointTypes.wt_conv_bias_t).toLong
      )
      val scale_fixed = scale_float.map(f => 
        FixedPointTypes.floatToFixed(f, FixedPointTypes.wt_conv_ws_t).toLong
      )
      
      println("Loading reference output...")
      val ref_file = s"${utils.DataPathConfig.data_path_prefix}/ref_float32_block/layers.${LAYER_IDX}.mixer.x_proj.input.float32.bin"
      val ref_output_float = read_float_file(ref_file, NUM_PATCHES * D_INNER)
      val ref_output = ref_output_float.map(f => 
        FixedPointTypes.floatToFixed(f, FixedPointTypes.fm_t).toLong
      )
      
      println("Writing data to AXI memory...")
      val AXI_XFER_BIT_WIDTH_VAL = 256
      val FEATURE_BLOCK_SIZE = 8
      val NUM_FEATURE_BLOCKS_INNER = (D_INNER + FEATURE_BLOCK_SIZE - 1) / FEATURE_BLOCK_SIZE
      val total_blocks = NUM_PATCHES * NUM_FEATURE_BLOCKS_INNER
      
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
      
      write_int4_array_to_axi(axi_mem_weights, dut.clockDomain, addr_weights_mags, weights_mags_int4, total_weight_elems, AXI_XFER_BIT_WIDTH_VAL, "Weight Mags")
      write_bit_array_to_axi(axi_mem_weights, dut.clockDomain, addr_weights_signs, weights_signs_bit, total_weight_elems, AXI_XFER_BIT_WIDTH_VAL, "Weight Signs")
      
      // CONV uses CONV_BLOCK_SIZE=16, which requires 2 consecutive words per block (8 elements each)
      // Write sequentially: word0[elem 0-7], word1[elem 8-15], word2[elem 16-23], ...
      val ELEMENTS_PER_WORD = AXI_XFER_BIT_WIDTH_VAL / 32 // 8
      val total_bias_words = (D_INNER + ELEMENTS_PER_WORD - 1) / ELEMENTS_PER_WORD
      
      for (w <- 0 until total_bias_words) {
        val word_base_addr = (addr_bias + w * (AXI_XFER_BIT_WIDTH_VAL / 8)).toLong
        
        for (j <- 0 until ELEMENTS_PER_WORD) {
          val idx = w * ELEMENTS_PER_WORD + j
          if (idx < D_INNER) {
            val value = BigInt(bias_fixed(idx))
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
      
      // Pack weight scales the same way - sequential words
      val total_scale_words = (D_INNER + ELEMENTS_PER_WORD - 1) / ELEMENTS_PER_WORD
      
      for (w <- 0 until total_scale_words) {
        val word_base_addr = (addr_scales + w * (AXI_XFER_BIT_WIDTH_VAL / 8)).toLong
        
        for (j <- 0 until ELEMENTS_PER_WORD) {
          val idx = w * ELEMENTS_PER_WORD + j
          if (idx < D_INNER) {
            val value = BigInt(scale_fixed(idx))
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
      
      // Wait for all memory writes to complete before reset
      dut.clockDomain.waitSampling(10)
      
      // Reset AXI memory simulators
      axi_mem_in.reset()
      axi_mem_out.reset()
      axi_mem_weights.reset()
      axilite_driver.reset()
      axilite_driver_r.reset()
      
      // Wait longer after reset for AXI-Lite to be ready
      println("Waiting after reset for interfaces to stabilize...")
      dut.clockDomain.waitSampling(100)
      
      println("Configuring module via AXI-Lite...")
      
      // Write memory addresses to s_axi_control_r FIRST
      // Match LINEAR.scala exactly - no CTRL register writes (they're reserved addresses)
      println("Writing memory addresses to s_axi_control_r...")
      // dst: 0x10 (lower), 0x14 (upper)
      axilite_driver_r.write(0x10, (addr_dst & 0xFFFFFFFFL).toInt)
      axilite_driver_r.write(0x14, ((addr_dst >> 32) & 0xFFFFFFFFL).toInt)
      
      // src: 0x1c (lower), 0x20 (upper)
      axilite_driver_r.write(0x1c, (addr_src & 0xFFFFFFFFL).toInt)
      axilite_driver_r.write(0x20, ((addr_src >> 32) & 0xFFFFFFFFL).toInt)
      
      // weights_mags_base: 0x28 (lower), 0x2c (upper)
      axilite_driver_r.write(0x28, (addr_weights_mags & 0xFFFFFFFFL).toInt)
      axilite_driver_r.write(0x2c, ((addr_weights_mags >> 32) & 0xFFFFFFFFL).toInt)
      
      // weights_signs_base: 0x34 (lower), 0x38 (upper)
      axilite_driver_r.write(0x34, (addr_weights_signs & 0xFFFFFFFFL).toInt)
      axilite_driver_r.write(0x38, ((addr_weights_signs >> 32) & 0xFFFFFFFFL).toInt)
      
      // bias_base: 0x40 (lower), 0x44 (upper)
      axilite_driver_r.write(0x40, (addr_bias & 0xFFFFFFFFL).toInt)
      axilite_driver_r.write(0x44, ((addr_bias >> 32) & 0xFFFFFFFFL).toInt)
      
      // weight_scales_base: 0x4c (lower), 0x50 (upper)
      axilite_driver_r.write(0x4c, (addr_scales & 0xFFFFFFFFL).toInt)
      axilite_driver_r.write(0x50, ((addr_scales >> 32) & 0xFFFFFFFFL).toInt)
      
      // Write scalar parameters to s_axi_control
      println("Writing scalar parameters to s_axi_control...")
      axilite_driver.write(ADDR_CONV_DIM, D_INNER)
      axilite_driver.write(ADDR_NUM_PATCHES, NUM_PATCHES)
      
      // Start computation (set AP_START bit and AP_CONTINUE for ap_ctrl_chain)
      // Use the exact same sequence as LINEAR.scala which works
      println("Starting computation...")
      // For ap_ctrl_chain protocol:
      // Bit 0: AP_START
      // Bit 4: AP_CONTINUE (must be set for ap_ctrl_chain)
      axilite_driver.write(ADDR_AP_CTRL, 0x11) // AP_START=1, AP_CONTINUE=1
      dut.clockDomain.waitSampling()
      axilite_driver.write(ADDR_AP_CTRL, 0x10) // Clear AP_START, keep AP_CONTINUE=1
      
      // Wait for completion (poll AP_DONE)
      println("Waiting for completion...")
      var done = false
      var cycles = 0
      val max_cycles = 1000000
      while (!done && cycles < max_cycles) {
        val ctrl = axilite_driver.read(ADDR_AP_CTRL)
        done = (ctrl & 2) != 0 // AP_DONE bit
        val ap_idle = (ctrl & 4) != 0 // AP_IDLE bit
        val ap_ready = (ctrl & 8) != 0 // AP_READY bit
        if (cycles == 0 || cycles == 1000 || cycles == 10000 || (cycles % 50000 == 0 && cycles > 10000)) {
          println(s"Cycle $cycles: AP_CTRL=0x${ctrl.toString(16)} (DONE=$done, IDLE=$ap_idle, READY=$ap_ready)")
          println(s"  Memory access: IN_R reads(req/resp)=$read_req_count_in/$read_resp_count_in, OUT_R writes(req/resp)=$write_req_count_out/$write_resp_count_out, WEIGHTS reads(req/resp)=$read_req_count_weights/$read_resp_count_weights")
        }
        if (!done) {
          dut.clockDomain.waitSampling(100)
          cycles += 100
        }
      }
      
      // Final memory access summary
      println(s"\nFinal Memory Access Summary:")
      println(s"  IN_R: $read_req_count_in read requests, $read_resp_count_in read responses")
      println(s"  OUT_R: $write_req_count_out write requests, $write_resp_count_out write responses")
      println(s"  WEIGHTS: $read_req_count_weights read requests, $read_resp_count_weights read responses")
      
      if (!done) {
        println(s"ERROR: Computation did not complete within $max_cycles cycles")
        simFailure()
      } else {
        println(s"Computation completed in $cycles cycles")
      }
      
      println("Reading results from AXI memory...")
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
      
      val dut_output = dut_output_flat
      
      println("Comparing results...")
      compare_arrays(ref_output, dut_output, "CONV Output", FixedPointTypes.fm_t)
      
      simSuccess()
      
      dut.clockDomain.waitSampling(100)
    }
}

