package utils

case class FixedPointType(
  totalBits: Int,
  integerBits: Int,
  isSigned: Boolean
) {
  val fractionalBits: Int = totalBits - integerBits
  val scale: Long = 1L << fractionalBits // 2^fractionalBits
  val maxValue: Long = if (isSigned) {
    (1L << (totalBits - 1)) - 1
  } else {
    (1L << totalBits) - 1
  }
  val minValue: Long = if (isSigned) {
    -(1L << (totalBits - 1))
  } else {
    0L
  }
}

object FixedPointTypes {
  
  // Feature map types (ap_fixed<32, 14>)
  val fm_t = FixedPointType(32, 14, isSigned = true)
  val gate_t = FixedPointType(32, 14, isSigned = true)
  val token_t = FixedPointType(32, 14, isSigned = true)
  val scan_t = FixedPointType(32, 14, isSigned = true)
  
  // Pixel type (ap_fixed<32, 10>)
  val pixel_t = FixedPointType(32, 10, isSigned = true)
  
  // Patch embedding types
  val wt_patch_embed_t = FixedPointType(16, 1, isSigned = true)
  val wt_patch_bias_t = FixedPointType(16, 4, isSigned = true)
  
  // RMS Norm types
  val wt_norm_t = FixedPointType(16, 1, isSigned = true)
  
  // Linear Layer types
  val wt_linear_bias_t = FixedPointType(32, 14, isSigned = true)
  val wt_linear_ws_t = FixedPointType(32, 6, isSigned = false)  // ap_ufixed<32, 6>
  val wt_linear_as_t = FixedPointType(32, 14, isSigned = false) // ap_ufixed<32, 14>
  val wt_linear_ss_t = FixedPointType(32, 6, isSigned = false)  // ap_ufixed<32, 6>
  
  // Conv Layer types
  val wt_conv_bias_t = FixedPointType(32, 14, isSigned = true)
  val wt_conv_ws_t = FixedPointType(32, 6, isSigned = false)   // ap_ufixed<32, 6>
  val wt_conv_as_t = FixedPointType(32, 14, isSigned = false)  // ap_ufixed<32, 14>
  
  /**
   * Convert float32 to fixed-point using the specified type
   * @param f Float value
   * @param fpType Fixed-point type configuration
   * @return BigInt representing the fixed-point value
   */
  def floatToFixed(f: Float, fpType: FixedPointType): BigInt = {
    val scaled = (f * fpType.scale).toLong // Use truncation, not rounding
    
    if (fpType.isSigned) {
      // Clamp to signed range
      val clamped = if (scaled > fpType.maxValue) fpType.maxValue
                   else if (scaled < fpType.minValue) fpType.minValue
                   else scaled
      BigInt(clamped)
    } else {
      // Clamp to unsigned range
      val clamped = if (scaled < 0) 0L
                   else if (scaled > fpType.maxValue) fpType.maxValue
                   else scaled
      BigInt(clamped)
    }
  }
  
  def fixedToFloat(b: BigInt, fpType: FixedPointType): Float = {
    val long_val = b.toLong
    
    val signed_val = if (fpType.isSigned) {
      // Handle two's complement for signed types
      if (long_val >= (1L << (fpType.totalBits - 1))) {
        long_val - (1L << fpType.totalBits)
      } else {
        long_val
      }
    } else {
      // Unsigned types
      if (long_val < 0) 0L
      else if (long_val > fpType.maxValue) fpType.maxValue
      else long_val
    }
    
    signed_val.toFloat / fpType.scale.toFloat
  }
}

