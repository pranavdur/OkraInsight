//
//  UIImage+MLMultiArray.swift
//  Vision + CoreML
//
//  Created by Ananth Durbha on 11/29/23.
//  Copyright © 2023 Apple. All rights reserved.
//

import CoreML

// Usage:
//     let mlMultiArray:MLMultiArray = uiImage.mlMultiArray()
//
// or if you need preprocess ...
//     let preProcessedMlMultiArray:MLMultiArray = uiImage.mlMultiArray(scale: 127.5, rBias: -1, gBias: -1, bBias: -1)
//
// or if you have gray scale image ...
//     let grayScaleMlMultiArray:MLMultiArray = uiImage.mlMultiArrayGrayScale()
extension UIImage {
    func toMLShapedArrayNCWH() -> MLShapedArray<Float> {
        guard let cgImage = self.cgImage else {
            return MLShapedArray<Float>()
        }
        let w = cgImage.width
        let h = cgImage.height
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * w
        let bitsPerComponent = 8
        var rawBytes: [UInt8] = [UInt8](repeating: 0, count: w * h * 4)
        rawBytes.withUnsafeMutableBytes { ptr in
            if let cgImage = self.cgImage,
                let context = CGContext(data: ptr.baseAddress,
                                        width: w,
                                        height: h,
                                        bitsPerComponent: bitsPerComponent,
                                        bytesPerRow: bytesPerRow,
                                        space: CGColorSpaceCreateDeviceRGB(),
                                        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) {
                let rect = CGRect(x: 0, y: 0, width: w, height: h)
                context.draw(cgImage, in: rect)
            }
        }

        let mlarr_ncwh = try! MLMultiArray(shape: [1, 3, NSNumber(value: Float(w)), NSNumber(value: Float(h))], dataType: MLMultiArrayDataType.float)
        
        var inputShapedArray = MLShapedArray<Float>(mlarr_ncwh)
        
        let n = 0
        for i in 0..<w {
            for j in 0..<h {
                inputShapedArray[scalarAt: [n, 0, i, j]] = (Float32(rawBytes[j*4*w + i * 4  + 0])/255.0) // R
                inputShapedArray[scalarAt: [n, 1, i, j]] = (Float32(rawBytes[j*4*w + i * 4  + 1])/255.0) // G
                inputShapedArray[scalarAt: [n, 2, i, j]] = (Float32(rawBytes[j*4*w + i * 4  + 2])/255.0) // B
            }
        }

        return inputShapedArray
    }
}
