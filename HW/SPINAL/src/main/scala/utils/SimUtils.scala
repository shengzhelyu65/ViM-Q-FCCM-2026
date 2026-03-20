package utils

import spinal.core._
import spinal.core.sim._
import spinal.lib.bus.amba4.axilite.AxiLite4
import spinal.lib.bus.amba4.axis.Axi4Stream.Axi4Stream
import spinal.lib.bus.amba4.axi.sim.{AxiMemorySim, AxiMemorySimConfig}

import java.io.{DataInputStream, File, FileInputStream, FileOutputStream, PrintStream}
import java.nio.{ByteBuffer, ByteOrder}
import scala.math.Numeric.Implicits.infixNumericOps
import scala.reflect.ClassTag
import java.nio.file.Paths
import java.nio.file.Files

object init_clock {
  def apply(clockDomain: ClockDomain, period: Int): Unit = {
    clockDomain.forkStimulus(period = period)
    clockDomain.assertReset()
    clockDomain.waitSampling(5)
    clockDomain.deassertReset()
  }
}

// Common data path prefix - read from environment variable VIM_Q_HW_ROOT and append /data
// Usage: export VIM_Q_HW_ROOT=/path/to/ViM_Q/HW
object DataPathConfig {
  val data_path_prefix: String = {
    val hwRoot = sys.env.get("VIM_Q_HW_ROOT").getOrElse(
      sys.error("VIM_Q_HW_ROOT environment variable is not set. Please export VIM_Q_HW_ROOT=/path/to/ViM_Q/HW")
    )
    new File(hwRoot, "data").getAbsolutePath
  }
}

case class AddressSeparableAxiLite4Driver(axi: AxiLite4, clockDomain: ClockDomain) {

  def reset(): Unit = {
    axi.aw.valid #= false
    axi.w.valid #= false
    axi.ar.valid #= false
    axi.r.ready #= true
    axi.b.ready #= true
  }

  def read(address: BigInt): BigInt = {
    axi.ar.payload.prot.assignBigInt(6)
    axi.ar.valid #= true
    axi.ar.payload.addr #= address
    axi.r.ready #= true
    clockDomain.waitSamplingWhere(axi.ar.ready.toBoolean)

    axi.ar.valid #= false
    clockDomain.waitSamplingWhere(axi.r.valid.toBoolean)

    axi.r.ready #= false
    axi.r.payload.data.toBigInt
  }

  def write(address: BigInt, data: BigInt): Unit = {
    axi.aw.payload.prot.assignBigInt(6)
    axi.w.payload.strb.assignBigInt((1 << axi.w.payload.strb.getWidth) - 1)
    val aw_thread = fork {
      axi.aw.valid #= true
      axi.aw.payload.addr #= address
      clockDomain.waitSamplingWhere(axi.aw.ready.toBoolean)
      axi.aw.valid #= false
    }
    val w_thread = fork {
      axi.w.valid #= true
      axi.w.payload.data #= data
      clockDomain.waitSamplingWhere(axi.w.ready.toBoolean)
      axi.w.valid #= false
    }
    aw_thread.join()
    w_thread.join()
  } // end of write
}

object init_daisy_chain {
  def apply(daisy_chain: DaisyChain[ManagerSignals]): Unit = {
    daisy_chain.I.L_BEGIN.randomize()
    daisy_chain.I.L_CLOSE.randomize()
    daisy_chain.I.POS.randomize()
    daisy_chain.I.T #= false
  }
}

// ============================================================================
// Float32 and Fixed-Point Conversion
// ============================================================================
object read_float_file {
  def apply(file: String, n: Int): Array[Float] = {
    val file_stream = new DataInputStream(new FileInputStream(file))
    val byte_buffer = ByteBuffer.allocate(4 * n) // 4 bytes per float32
    file_stream.read(byte_buffer.array())
    file_stream.close()
    val float_buffer = byte_buffer.order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer() // little endian
    val float_array = new Array[Float](n)
    float_buffer.get(float_array)
    float_array
  }
}

object read_int4_file {
  def apply(file: String, n: Int): Array[Int] = {
    val file_stream = new DataInputStream(new FileInputStream(file))
    val bytes_needed = (n + 1) / 2 // Read ceil(n/2) bytes for n int4 values
    val byte_buffer = ByteBuffer.allocate(bytes_needed)
    file_stream.read(byte_buffer.array())
    file_stream.close()
    val bytes = byte_buffer.array()
    val int4_array = new Array[Int](n)
    
    // Each byte contains TWO int4 values:
    // - Upper nibble (bits 7-4): first int4
    // - Lower nibble (bits 3-0): second int4
    for (i <- 0 until n) {
      val byte_idx = i / 2
      val is_upper_nibble = (i % 2) == 0
      if (byte_idx < bytes.length) {
        val byte_val = bytes(byte_idx) & 0xFF
        if (is_upper_nibble) {
          int4_array(i) = (byte_val >> 4) & 0x0F // Upper nibble
        } else {
          int4_array(i) = byte_val & 0x0F // Lower nibble
        }
      } else {
        int4_array(i) = 0
      }
    }
    int4_array
  }
}

object read_bit_file {
  def apply(file: String, n: Int): Array[Int] = {
    val file_stream = new DataInputStream(new FileInputStream(file))
    val bytes_needed = (n + 7) / 8 // Read ceil(n/8) bytes
    val byte_buffer = ByteBuffer.allocate(bytes_needed)
    file_stream.read(byte_buffer.array())
    file_stream.close()
    val bytes = byte_buffer.array()
    val bit_array = new Array[Int](n)
    
    // Each byte contains 8 bits
    // Assuming LSB first (bit 0 is first element) to match typical packing
    for (i <- 0 until n) {
      val byte_idx = i / 8
      val bit_idx = i % 8
      if (byte_idx < bytes.length) {
        val byte_val = bytes(byte_idx) & 0xFF
        bit_array(i) = (byte_val >> bit_idx) & 0x01
      } else {
        bit_array(i) = 0
      }
    }
    bit_array
  }
}

// ============================================================================
// AXI Memory Writing Functions
// ============================================================================

object write_int4_array_to_axi {
  def apply(
    axi_mem: AxiMemorySim,
    clockDomain: ClockDomain,
    base_addr: BigInt,
    data: Array[Int],
    total_elements: Int,
    axi_data_width: Int,
    info: String
  ): Unit = {
    
    val int4_per_word = axi_data_width / 4 // 64 int4 per 256-bit word
    val bytes_per_word = axi_data_width / 8 // 32 bytes per 256-bit word
    val total_words = (total_elements + int4_per_word - 1) / int4_per_word
    
    // The data array should already contain unpacked int4 values (0-15)
    // We pack them into 256-bit words: each int4 takes 4 bits, 64 int4 per word
    for (w <- 0 until total_words) {
      var word = BigInt(0)
      for (t <- 0 until int4_per_word) {
        val idx = w * int4_per_word + t
        if (idx < total_elements && idx < data.length) {
          val int4_val = data(idx) & 0x0F // Ensure 4-bit value (0-15)
          word = word | (BigInt(int4_val) << (4 * t))
        }
      }
      val addr = (base_addr + w * bytes_per_word).toLong
      axi_mem.memory.writeBigInt(addr, word, bytes_per_word)
    }
  }
}

object write_bit_array_to_axi {
  def apply(
  axi_mem: AxiMemorySim,
  clockDomain: ClockDomain,
  base_addr: BigInt,
  data: Array[Int],
  total_elements: Int,
  axi_data_width: Int,
  info: String
): Unit = {
  val bits_per_word = axi_data_width // 256 bits per 256-bit word
  val bytes_per_word = axi_data_width / 8 // 32 bytes per 256-bit word
  val total_words = (total_elements + bits_per_word - 1) / bits_per_word
  
  for (w <- 0 until total_words) {
    var word = BigInt(0)
    for (b <- 0 until bits_per_word) {
      val idx = w * bits_per_word + b
      if (idx < total_elements && idx < data.length) {
        val bit_val = data(idx) & 0x01 // Ensure 1-bit value
        word = word | (BigInt(bit_val) << b)
      }
    }
      val addr = (base_addr + w * bytes_per_word).toLong
      axi_mem.memory.writeBigInt(addr, word, bytes_per_word)
    }
  }
}

object compare_arrays {
  def apply(
    ref_array: Array[Long],
    dut_array: Array[Long],
    test_name: String,
    fpType: FixedPointType = FixedPointTypes.fm_t
  ): Unit = {
    require(ref_array.length == dut_array.length, s"Array length mismatch: ref=${ref_array.length}, dut=${dut_array.length}")
    
    var mse = 0.0
    var mae = 0.0
    var max_error = 0.0
    var error_count = 0
    
    for (i <- ref_array.indices) {
      val ref_val = FixedPointTypes.fixedToFloat(BigInt(ref_array(i)), fpType)
      val dut_val = FixedPointTypes.fixedToFloat(BigInt(dut_array(i)), fpType)
      val error = scala.math.abs(ref_val - dut_val)
      val rel_error = if (ref_val != 0) error / scala.math.abs(ref_val) else error
      max_error = scala.math.max(max_error, error)
      mse += error * error
      mae += error
      if (error > 0.01 || rel_error > 0.01) { // Threshold for reporting
        error_count += 1
        if (error_count <= 10) { // Only print first 10 errors
          println(f"  Error at $i: ref=$ref_val%.6f, dut=$dut_val%.6f, error=$error%.6f")
        }
      }
    }
    
    val mean_mse = mse / ref_array.length
    val mean_mae = mae / ref_array.length
    val rmse = scala.math.sqrt(mean_mse)
    
    // Print in format matching HLS output: MSE, MAE, RMSE, Max Error
    println(f"$test_name: MSE=$mean_mse%.6f, MAE=$mean_mae%.6f, RMSE=$rmse%.6f, Max Error=$max_error%.6f, Error Count=$error_count/${ref_array.length}")
    
    if (rmse < 0.5 && mean_mae < 1.0) {
      println(s"✓ $test_name: PASSED")
    } else {
      println(s"✗ $test_name: FAILED")
    }
  }
}