import Foundation

/// The numeric contract's operations, in Swift.
///
/// Every one of these reproduces `tools/ordered_reference.py` **bit for bit**, and the
/// tests assert bit patterns rather than closeness. Three rules make that possible, and
/// all three are load-bearing:
///
/// 1. **Ascending accumulation, one operation per step.** No split-K, no reassociation.
/// 2. **No fused multiply-add.** Swift does not contract `a * b + c` into an FMA by
///    default, which is what the contract needs; a compiler flag that turned it on would
///    silently break the gate, so the tests exist to catch exactly that.
/// 3. **Transcendentals in double precision, rounded to `Float`.** Measured: numpy's
///    float32 `sin`/`cos` differ from Swift's in 12–18% of cases, and Swift's differ from
///    libm's `sinf`/`cosf` in ~1%, while computing in double and rounding agrees in 0 of
///    6000 samples. `exp` agrees either way and is stated the same way so the rule is one
///    sentence instead of a table of exceptions.
public enum Ops {
    /// `x @ wᵀ`, `x` being `rows × k` and `w` being `out × k`.
    ///
    /// The inner loop is ordered by `k` and rounds after every multiply and add, which is
    /// the entire content of the contract: `PyTorch`'s matmul order is a property of its
    /// BLAS kernels and is not reproducible by any engine (measured: 3544 of 4096 outputs
    /// differ on a real layer's shapes, both equally close to fp64).
    public static func orderedMatmul(x: [Float], w: [Float], rows: Int, k: Int, out: Int) -> [Float] {
        // The scalar body below is the **definition**; this is the fast path, and it is bit-equal
        // to it on every shape the tests try (measured, `D9`). Keeping both is deliberate: a
        // faster formulation is only trustworthy while something independent says it agrees.
        orderedMatmulVectorized(x: x, w: w, rows: rows, k: k, out: out)
    }

    /// The definition: ascending accumulation over `k`, one rounding per multiply and per add.
    public static func orderedMatmulScalar(x: [Float], w: [Float], rows: Int, k: Int, out: Int) -> [Float] {
        var result = [Float](repeating: 0, count: rows * out)
        for row in 0..<rows {
            let xRow = row * k
            let rRow = row * out
            for column in 0..<out {
                let wRow = column * k
                var accumulator: Float = 0
                for index in 0..<k {
                    accumulator = accumulator + x[xRow + index] * w[wRow + index]
                }
                result[rRow + column] = accumulator
            }
        }
        return result
    }


    /// `x @ wᵀ` with four outputs at a time, and the **same additions in the same order**.
    ///
    /// The vector is taken across the **output** dimension, never across `k`. That is the whole
    /// trick: each lane accumulates over `k` ascending, one rounding per multiply and one per add,
    /// exactly as the scalar op does — so the result is bit-identical to the contract rather than
    /// merely close to it. Vectorising across `k` instead would need a horizontal reduction, which
    /// reassociates the sum and changes the bits.
    ///
    /// The product is materialised before the add (`accumulator + value * lane`) because the
    /// contract forbids fusing the two: `Float.addingProduct` is a different number, and the
    /// contract is defined by the rounding sequence rather than by the algebra.
    ///
    /// The tail is the scalar loop verbatim, because there is nothing to vectorise and a second
    /// formulation is a second chance to disagree.
    public static func orderedMatmulVectorized(x: [Float], w: [Float], rows: Int, k: Int, out: Int) -> [Float] {
        var result = [Float](repeating: 0, count: rows * out)
        let width = 4
        for row in 0..<rows {
            let xRow = row * k
            let rRow = row * out
            var column = 0
            while column + width <= out {
                var accumulator = SIMD4<Float>(repeating: 0)
                for index in 0..<k {
                    let value = SIMD4<Float>(repeating: x[xRow + index])
                    let weights = SIMD4<Float>(
                        w[(column + 0) * k + index],
                        w[(column + 1) * k + index],
                        w[(column + 2) * k + index],
                        w[(column + 3) * k + index]
                    )
                    accumulator = accumulator + value * weights
                }
                result[rRow + column + 0] = accumulator[0]
                result[rRow + column + 1] = accumulator[1]
                result[rRow + column + 2] = accumulator[2]
                result[rRow + column + 3] = accumulator[3]
                column += width
            }
            while column < out {
                let wRow = column * k
                var accumulator: Float = 0
                for index in 0..<k {
                    accumulator = accumulator + x[xRow + index] * w[wRow + index]
                }
                result[rRow + column] = accumulator
                column += 1
            }
        }
        return result
    }

    /// Ascending-index sum, one addition per step.
    public static func orderedSum(_ values: [Float]) -> Float {
        var total: Float = 0
        for value in values { total = total + value }
        return total
    }

    /// `x` is `rows × width`; the mean of squares is an ordered sum divided by `width`,
    /// and `1/sqrt` is IEEE-exact in both languages.
    public static func rmsNorm(x: [Float], weight: [Float], rows: Int, width: Int, eps: Float) -> [Float] {
        var result = [Float](repeating: 0, count: rows * width)
        let scale = Float(width)
        for row in 0..<rows {
            let base = row * width
            var squares: [Float] = []
            squares.reserveCapacity(width)
            for index in 0..<width {
                let value = x[base + index]
                squares.append(value * value)
            }
            let variance = orderedSum(squares) / scale
            let inverse = Float(1) / (variance + eps).squareRoot()
            for index in 0..<width {
                result[base + index] = weight[index] * (x[base + index] * inverse)
            }
        }
        return result
    }

    /// `exp` in double precision, rounded to `Float` — the contract's designation.
    @inline(__always)
    public static func exp32(_ x: Float) -> Float {
        Float(exp(Double(x)))
    }

    /// Stable sigmoid; which branch is taken is part of the contract, not an optimisation.
    public static func sigmoid(_ x: Float) -> Float {
        if x >= 0 {
            return Float(1) / (Float(1) + exp32(-x))
        }
        let exponential = exp32(x)
        return exponential / (Float(1) + exponential)
    }

    /// `x * sigmoid(x)`.
    public static func silu(_ x: Float) -> Float {
        x * sigmoid(x)
    }

    public static func silu(_ x: [Float]) -> [Float] {
        x.map(silu)
    }

    /// Row-wise softmax: max-subtracted, ordered sum, divide.
    public static func softmax(x: [Float], rows: Int, width: Int) -> [Float] {
        var result = [Float](repeating: 0, count: rows * width)
        for row in 0..<rows {
            let base = row * width
            var maximum = x[base]
            for index in 1..<width where x[base + index] > maximum { maximum = x[base + index] }
            var exponentials: [Float] = []
            exponentials.reserveCapacity(width)
            for index in 0..<width { exponentials.append(exp32(x[base + index] - maximum)) }
            let total = orderedSum(exponentials)
            for index in 0..<width { result[base + index] = exponentials[index] / total }
        }
        return result
    }

    /// Rotate-half tables: `cos` and `sin` of the doubled angle, in double then rounded.
    ///
    /// The engine computes these on the CPU even when the matmuls run on the GPU. Apple
    /// GPUs have no fp64, so a Metal kernel could not reproduce this rule; the tables
    /// depend only on position and head width, so they are computed once and passed down
    /// as a buffer.
    public static func ropeTables(headDim: Int, positions: [Double], theta: Double) -> (cos: [Float], sin: [Float]) {
        let theta32 = Float(theta)
        var frequencies: [Float] = []
        frequencies.reserveCapacity(headDim / 2)
        for index in stride(from: 0, to: headDim, by: 2) {
            let exponent = Float(index) / Float(headDim)
            frequencies.append(Float(1) / powf(theta32, exponent))
        }
        var cosines = [Float](repeating: 0, count: positions.count * headDim)
        var sines = [Float](repeating: 0, count: positions.count * headDim)
        for (row, position) in positions.enumerated() {
            for index in 0..<headDim {
                let angle = position * Double(frequencies[index % frequencies.count])
                cosines[row * headDim + index] = Float(cos(angle))
                sines[row * headDim + index] = Float(sin(angle))
            }
        }
        return (cosines, sines)
    }

    /// Rotate-half RoPE: element `i` pairs with element `i + headDim/2`, each product
    /// rounded before the sum.
    public static func applyRope(
        x: [Float], cos cosines: [Float], sin sines: [Float], tokens: Int, heads: Int, headDim: Int
    ) -> [Float] {
        let half = headDim / 2
        var result = [Float](repeating: 0, count: tokens * heads * headDim)
        for token in 0..<tokens {
            for head in 0..<heads {
                let base = (token * heads + head) * headDim
                let table = token * headDim
                for index in 0..<half {
                    let first = x[base + index]
                    let second = x[base + index + half]
                    result[base + index] = first * cosines[table + index] + (-second) * sines[table + index]
                    result[base + index + half] = second * cosines[table + index + half] + first * sines[table + index + half]
                }
            }
        }
        return result
    }
}
