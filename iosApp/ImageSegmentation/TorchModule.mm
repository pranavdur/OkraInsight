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

- (void)testLayoutAndOrientation:(unsigned char*)buffer withWidth:(int)width withHeight:(int)height {
    for (int w = 0; w < width; w++) {
        for (int h = 0; h < height; h++) {
            int n = 3 * (h * width + w);
            buffer[n] = 0; buffer[n+1] = 0; buffer[n+2] = 0;
            if (w > 50 && w < 150)
                buffer[n] = 255;
            if (h > 250 && h < 350)
                buffer[n+1] = 255;
            
        }
    }
    return;
}

- (uint32_t)convertPixelCoordToIndex:(uint32_t)width withPixelX:(uint32_t)x withPixelY:(int)y {
    int n = 3 * (y * width + x);
    return n;
}

- (void)displayPixelColor:(unsigned char*)buffer withIndex:(uint32_t)index withR:(int)r withG:(int)g withB:(int)b {
    buffer[index] = r; buffer[index+1] = g; buffer[index+2] = b;
    return;
}

- (unsigned char*)segmentImage:(void *)imageBuffer withWidth:(int)width withHeight:(int)height {
    try {
        
        // see http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2007/segexamples/index.html for the list of classes with indexes
        // const int CLASSNUM = 21;
        // const int DOG = 12;
        // const int PERSON = 15;
        // const int SHEEP = 17;
        const int CLASSNUM = 1;
        const int OKRA = 0;

        at::Tensor tensor = torch::from_blob(imageBuffer, { 1, 3, width, height }, at::kFloat);

        float* floatInput = tensor.data_ptr<float>();
        if (!floatInput) {
            return nil;
        }
        NSMutableArray* inputs = [[NSMutableArray alloc] init];
        for (int i = 0; i < 3 * width * height; i++) {
            [inputs addObject:@(floatInput[i])];
        }

        c10::InferenceMode guard;
        
        CFTimeInterval startTime = CACurrentMediaTime();
        // auto outputDict = _impl.forward({ tensor }).toGenericDict();
        // auto outputTensor = outputDict.at("out").toTensor();
        
        //void(^tuple)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor) = _impl.forward({ tensor });
        
        auto outputTensor = _impl.forward({ tensor }).toTuple()->elements()[0].toTensor();
        
        //at::Tensor outputTensor, d1, d2, d3, d4, d5, d6;
        //tuple(outputTensor, d1, d2, d3, d4, d5, d6);
        
        CFTimeInterval elapsedTime = CACurrentMediaTime() - startTime;
        NSLog(@"inference time:%f", elapsedTime);
        
 
        // std::vector<at::IValue> elements = outputTuple->elements();
        // auto outputTensor = elements[0].toTensor();
        float* floatBuffer = outputTensor.data_ptr<float>();
        if (!floatBuffer) {
            return nil;
        }
        NSMutableArray* results = [[NSMutableArray alloc] init];
        for (int i = 0; i < CLASSNUM * width * height; i++) {
            [results addObject:@(floatBuffer[i])];
        }

        NSMutableData* data = [NSMutableData dataWithLength:sizeof(unsigned char) * 3 * width * height];
        unsigned char* buffer = (unsigned char*)[data mutableBytes];
        
        // [self testLayoutAndOrientation:buffer withWidth:width withHeight:height];
        
        // go through each element in the output of size [WIDTH, HEIGHT] and
        // set different color for different classnum
        int r, g, b, a;
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < height; k++) {
                /*
                int maxi = 0, maxj = 0, maxk = 0;
                float maxnum = -100000.0;
                for (int i = 0; i < CLASSNUM; i++) {
                    if ([results[i * (width * height) + j * width + k] floatValue] > maxnum) {
                        maxnum = [results[i * (width * height) + j * width + k] floatValue];
                        maxi = i; maxj = j; maxk = k;
                    }
                }

                int n = 3 * (maxj * width + maxk);
                // color coding for person (red), dog (green), sheep (blue)
                // black color for background and other classes
                buffer[n] = 0; buffer[n+1] = 0; buffer[n+2] = 0;
                if (maxi == PERSON) buffer[n] = 255;
                else if (maxi == DOG) buffer[n+1] = 255;
                else if (maxi == SHEEP) buffer[n+2] = 255;
                */
                
                /*
                 // input tensor
                r = *((tensor[0][0][k][j]).data_ptr<float>()) * 255;
                g = *((tensor[0][1][k][j]).data_ptr<float>()) * 255;
                b = *((tensor[0][2][k][j]).data_ptr<float>()) * 255;
                */
                
                 // mask prediction
                Float32 mask = *((outputTensor[0][0][j][k]).data_ptr<Float32>());
                int32_t imask = static_cast<int32_t>(mask);
                
                
                r = imask * 255;
                g = (imask >> 8) * 255;
                b = (imask >> 16) * 255;
                a = (imask >> 24) * 255;
                
                
                  // OkraSurface pixels: (112,200) imask:0 a:0 r:0 g:0 b:0
                  // non-OkraSurface pixels: (118,200) imask:-1 a:255 r:255 g:255 b:255
                 
                if (j == 102)
                    NSLog(@"(%d,%d) imask:%d a:%d r:%d g:%d b:%d", j, k, imask, a, r,g,b);
                
                
               
                
                r = g = b = 255;
                
                if (imask == 1) {
                    r = *((tensor[0][0][j][k]).data_ptr<float>()) * 255;
                    g = *((tensor[0][1][j][k]).data_ptr<float>()) * 255;
                    b = *((tensor[0][2][j][k]).data_ptr<float>()) * 255;
                    
                }
                
                
                int n = [self convertPixelCoordToIndex:height withPixelX:k withPixelY:j];
                [self displayPixelColor:buffer withIndex:n withR:r withG:g withB:b];
            }
        }
        

        return buffer;
    } catch (const std::exception& exception) {
        NSLog(@"%s", exception.what());
    }
    return nil;
}

@end
