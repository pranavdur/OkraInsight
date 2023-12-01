//
//  UIImage+Helper.swift
//  Vision + CoreML
//
//  Created by Ananth Durbha on 11/19/23.
//  Copyright © 2023 Apple. All rights reserved.
//

import UIKit
import VideoToolbox
import CoreML

extension UIImage {
    func resized(to newSize: CGSize, scale: CGFloat = 1) -> UIImage {
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = scale
        let renderer = UIGraphicsImageRenderer(size: newSize, format: format)
        let image = renderer.image { _ in
            draw(in: CGRect(origin: .zero, size: newSize))
        }
        return image
    }

    func normalized() -> [Float32]? {
        guard let cgImage = self.cgImage else {
            return nil
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
        var normalizedBuffer: [Float32] = [Float32](repeating: 0, count: w * h * 3)
        // normalize the pixel buffer
        // see https://pytorch.org/hub/pytorch_vision_resnet/ for more detail
        
        // normalize = transforms.Normalize(mean=[0.25, 0.2, 0.406], std=[0.129, 0.124, 0.225])
        // normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        for i in 0 ..< w * h {
            normalizedBuffer[i] = (Float32(rawBytes[i * 4 + 0]) / 255.0 - 0.250) / 0.129 // R
            normalizedBuffer[w * h + i] = (Float32(rawBytes[i * 4 + 1]) / 255.0 - 0.200) / 0.124 // G
            normalizedBuffer[w * h * 2 + i] = (Float32(rawBytes[i * 4 + 2]) / 255.0 - 0.406) / 0.225 // B
        }
        return normalizedBuffer
    }
    
    func regularized() -> [Float32]? {
        guard let cgImage = self.cgImage else {
            return nil
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
        var normalizedBuffer: [Float32] = [Float32](repeating: 0, count: w * h * 3)
        // normalize the pixel buffer
        // see https://pytorch.org/hub/pytorch_vision_resnet/ for more detail
        for i in 0 ..< w * h {
            normalizedBuffer[i] = (Float32(rawBytes[i * 4 + 0]) / 255.0) // R
            normalizedBuffer[w * h + i] = (Float32(rawBytes[i * 4 + 1]) / 255.0) // G
            normalizedBuffer[w * h * 2 + i] = (Float32(rawBytes[i * 4 + 2]) / 255.0) // B
        }
        return normalizedBuffer
    }
    
    /// Get the pixel color at a point in the image
    func pixelColor(atLocation point: CGPoint) -> UIColor? {
        guard let cgImage = cgImage, let pixelData = cgImage.dataProvider?.data else {
            return nil
        }
        let data: UnsafePointer<UInt8> = CFDataGetBytePtr(pixelData)
        let bytesPerPixel = cgImage.bitsPerPixel / 8
        let pixelInfo: Int = ((cgImage.bytesPerRow * Int(point.y)) + (Int(point.x) * bytesPerPixel))
        let r = CGFloat(data[pixelInfo]) / CGFloat(255.0)
        let g = CGFloat(data[pixelInfo + 1]) / CGFloat(255.0)
        let b = CGFloat(data[pixelInfo + 2]) / CGFloat(255.0)
        let a = CGFloat(data[pixelInfo + 3]) / CGFloat(255.0)
        return UIColor(red: r, green: g, blue: b, alpha: a)
    }
    
    func dumpRaster() -> Void {
        guard let cgImage = self.cgImage else {
            return
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

        // normalize the pixel buffer
        // see https://pytorch.org/hub/pytorch_vision_resnet/ for more detail

        // normalize = transforms.Normalize(mean=[0.25, 0.2, 0.406], std=[0.129, 0.124, 0.225])
        // normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        for i in 0 ..< w {
            print("200,",i, ":" ,(Float32(rawBytes[200*4*w + i * 4  + 0])/255.0)) // R
            // print("200,",i, ":" ,(Float32(rawBytes[200*4*w + i * 4  + 1])/255.0)) // G
            // print("200,",i, ":" ,(Float32(rawBytes[200*4*w + i * 4  + 2])/255.0)) // B
        }
        return
    }
    
    
    func copyPerPixelColorToBuffer(buffer: UnsafeMutablePointer<UInt8>) -> Void {
        guard let cgImage = self.cgImage else {
            return
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
   
        for i in 0..<h {
            for j in 0..<w {
                let linearIndex   =  (i * w + j)
                let bytesPerBufferIndex = 3
                let bytesPerImageData = 4
                
                buffer[linearIndex * bytesPerBufferIndex + 0] = rawBytes[linearIndex * bytesPerImageData + 0]  // R
                buffer[linearIndex * bytesPerBufferIndex + 1] = rawBytes[linearIndex * bytesPerImageData + 1]  // G
                buffer[linearIndex * bytesPerBufferIndex + 2] = rawBytes[linearIndex * bytesPerImageData + 2]  // B
            }
        }
        return
    }
}
