// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#import "TorchModule.h"
#import "UIImageHelper.h"
#import <CoreImage/CoreImage.h>
#import <ImageIO/ImageIO.h>
#import <Libtorch/Libtorch.h>

@implementation TorchModule {
@protected
    torch::jit::script::Module _impl;
}

typedef void (*callbackType)(unsigned char* outputImgBuffer, at::Tensor& inputImgTensor, at::Tensor& okraSegMask, int width, int height, int w, int h, int& inpR, int& inpG, int& inpB);

static uint32_t convertPixelCoordToIndex(uint32_t width, uint32_t x, int y) {
    int n = 3 * (y * width + x);
    return n;
}

- (nullable instancetype)initWithFileAtPath:(NSString*)filePath {
    self = [super init];
    if (self) {
        try {
            _impl = torch::jit::load(filePath.UTF8String);
            _impl.eval();
        } catch (const std::exception& exception) {
            NSLog(@"%s", exception.what());
            return nil;
        }
    }
    return self;
}




static void updateOutputImgColor(unsigned char* outputImgBuffer, int width, int height, int w, int h, int& r, int& g, int& b) {
    int n = convertPixelCoordToIndex(width, w, h);
    outputImgBuffer[n] = r; outputImgBuffer[n+1] = g; outputImgBuffer[n+2] = b;
}

// assumes given tensor shape is NCHW.
static void copyColorFromInputImg(unsigned char* outputImgBuffer, at::Tensor& inputImgTensor, at::Tensor& okraSegMask /* Ignored */, int width, int height, int w, int h, int& inpR, int& inpG, int& inpB) {
    
    inpR = *((inputImgTensor[0][0][w][h]).data_ptr<float>()) * 255;
    inpG = *((inputImgTensor[0][1][w][h]).data_ptr<float>()) * 255;
    inpB = *((inputImgTensor[0][2][w][h]).data_ptr<float>()) * 255;
}

static void copyFromOkraSegMask(unsigned char* outputImgBuffer, at::Tensor& inputImgTensor, at::Tensor& okraSegMask, int width, int height, int w, int h, int& inpR, int& inpG, int& inpB) {
    
    Float32 probPixelIsOnOkra = *((okraSegMask[0][0][w][h]).data_ptr<Float32>());
    
    inpR = probPixelIsOnOkra * 255;
    inpG = 0;
    inpB = 0;

}

// assumes given tensor shape is NCWH.
static void copyOkraPixelsFromInputImg(unsigned char* outputImgBuffer, at::Tensor& inputImgTensor, at::Tensor& okraSegMask, int width, int height, int w, int h, int& inpR, int& inpG, int& inpB) {
    
    Float32 probPixelIsOnOkra = *((okraSegMask[0][0][w][h]).data_ptr<Float32>());
    
    // if (h == 200)
    //    NSLog(@"mask (%d, %d) : %f", w, h, probPixelIsOnOkra);
    
    if (probPixelIsOnOkra > 0.98) {
        inpR = *((inputImgTensor[0][0][w][h]).data_ptr<float>()) * 255;
        inpG = *((inputImgTensor[0][1][w][h]).data_ptr<float>()) * 255;
        inpB = *((inputImgTensor[0][2][w][h]).data_ptr<float>()) * 255;
    } else {
        // Blacken non-Okra part of the image
        inpR = 0;
        inpG = 0;
        inpB = 0;
    }

}

static void drawTestBand(unsigned char* outputImgBuffer, at::Tensor& inputImgTensor /* Ignored */, at::Tensor& okraSegMask /* Ignored */, int width, int height, int w, int h, int& inpR, int& inpG, int& inpB) {
    
    inpR = 0;
    inpG = 0;
    inpB = 0;
    

    if (w > 50 && w < 150)
        inpR = 255;
    if (h > 350 && h < 400)
        inpG = 255;
    
}

static void forEachInputAndOutputImgBufferElem (callbackType getPerElemColorFn, at::Tensor& inputImgTensor, unsigned char* outputImgBuffer, at::Tensor& okraSegMask, int width, int height) {
    
    int inpR = 0;
    int inpG = 0;
    int inpB = 0;
    
    
    // permuted shape: NCWH as needed by segmentation model
    // tensor = tensor.permute({0,1,3,2});
    
    for (int w = 0; w < width; w++) {
        for (int h = 0; h < height; h++) {
            getPerElemColorFn(outputImgBuffer, inputImgTensor, okraSegMask, width, height, w, h, inpR, inpG, inpB);
            updateOutputImgColor(outputImgBuffer, width, height, w, h, inpR, inpG, inpB);
        }
    }
}


static void testLayoutAndOrientation(at::Tensor& inputImgTensor, unsigned char* outputImgBuffer, at::Tensor& okraSegMask, int width, int height) {
    forEachInputAndOutputImgBufferElem(drawTestBand, inputImgTensor, outputImgBuffer, okraSegMask, width, height);
}
 
static void sanityTest1 (at::Tensor&  inputImgTensor, unsigned char* outputImgBuffer, at::Tensor& okraSegMask, int width, int height) {
    forEachInputAndOutputImgBufferElem(copyColorFromInputImg, inputImgTensor, outputImgBuffer, okraSegMask, width, height);
}

static void applyOkraSegmentMask (at::Tensor&  inputImgTensor, unsigned char* outputImgBuffer, at::Tensor& okraSegMask, int width, int height) {
    forEachInputAndOutputImgBufferElem(copyOkraPixelsFromInputImg, inputImgTensor, outputImgBuffer, okraSegMask, width, height);
}

- (unsigned char*)segmentImage:(void *)imageBuffer withWidth:(int)width withHeight:(int)height {
    try {
        
        // see http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2007/segexamples/index.html for the list of classes with indexes
        const int CLASSNUM = 1;
        const int OKRA = 0;

        // native shape: NCHW
        at::Tensor tensor = torch::from_blob(imageBuffer, { 1, 3, height, width }, at::kFloat);
        
        // permuted shape: NCWH as needed by segmentation model
        tensor = tensor.permute({0,1,3,2});
        
    
        c10::InferenceMode guard;
        
        CFTimeInterval startTime = CACurrentMediaTime();
        // auto outputDict = _impl.forward({ tensor }).toGenericDict();
        // auto outputTensor = outputDict.at("out").toTensor();
                
        auto outputTensor = _impl.forward({ tensor }).toTuple()->elements()[0].toTensor();
                
        CFTimeInterval elapsedTime = CACurrentMediaTime() - startTime;
        NSLog(@"inference time:%f", elapsedTime);
        
        NSMutableData* data = [NSMutableData dataWithLength:sizeof(unsigned char) * 3 * width * height];
        unsigned char* buffer = (unsigned char*)[data mutableBytes];
        

        // testLayoutAndOrientation(tensor, buffer, outputTensor, width, height);
        // sanityTest1(tensor, buffer, outputTensor, width, height);
        applyOkraSegmentMask(tensor, buffer, outputTensor, width, height);
    
        return buffer;
    } catch (const std::exception& exception) {
        NSLog(@"%s", exception.what());
    }
    return nil;
}

@end

