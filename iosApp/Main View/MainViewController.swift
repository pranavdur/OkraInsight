/*
See LICENSE folder for this sample’s licensing information.

Abstract:
The view controller that selects an image and makes a prediction using Vision and Core ML.
*/

import UIKit
import CoreML

class MainViewController: UIViewController {
    var firstRun = true
    private let imageHelper = UIImageHelper()
    /// A predictor instance that uses Vision and Core ML to generate prediction strings from a photo.
    let imagePredictor = ImagePredictor()

    /// The largest number of predictions the main view controller displays the user.
    let predictionsToShow = 2

    // MARK: Main storyboard outlets
    @IBOutlet weak var startupPrompts: UIStackView!
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var predictionLabel: UILabel!
}

extension MainViewController {
    // MARK: Main storyboard actions
    /// The method the storyboard calls when the user one-finger taps the screen.
    @IBAction func singleTap() {
        // Show options for the source picker only if the camera is available.
        guard UIImagePickerController.isSourceTypeAvailable(.camera) else {
            present(photoPicker, animated: false)
            return
        }

        present(cameraPicker, animated: false)
    }

    /// The method the storyboard calls when the user two-finger taps the screen.
    @IBAction func doubleTap() {
        present(photoPicker, animated: false)
    }
}

extension MainViewController {
    // MARK: Main storyboard updates
    /// Updates the storyboard's image view.
    /// - Parameter image: An image.
    func updateImage(_ image: UIImage) {
        DispatchQueue.main.async {
            self.imageView.image = image
        }
    }

    /// Updates the storyboard's prediction label.
    /// - Parameter message: A prediction or message string.
    /// - Tag: updatePredictionLabel
    func updatePredictionLabel(_ message: String) {
        DispatchQueue.main.async {
            self.predictionLabel.text = message
        }

        if firstRun {
            DispatchQueue.main.async {
                self.firstRun = false
                self.predictionLabel.superview?.isHidden = false
                self.startupPrompts.isHidden = true
            }
        }
    }
    /// Notifies the view controller when a user selects a photo in the camera picker or photo library picker.
    /// - Parameter photo: A photo from the camera or photo library.
    func userSelectedPhoto(_ photo: UIImage) {
        let resizedPhoto = photo.resized(to: CGSize(width: 302, height: 403))
        updateImage(resizedPhoto)
        updatePredictionLabel("Making predictions for the photo...")

        DispatchQueue.global(qos: .userInitiated).async {
            self.classifyImage(resizedPhoto)
        }
    }

}

// let shapedArray = MLShapedArray<Float>(data: imageData!, shape: [1, Int(image.size.height), Int(image.size.width), 4])

extension MainViewController {
    struct Prediction {
        let mask : MLShapedArray<Float>
    }
    
    // MARK: Image prediction methods
     /// Sends a photo to the Image Predictor to get a prediction of its content.
     /// - Parameter image: A photo.
     private func classifyImage(_ image: UIImage) {
         do {
             var predictions: [Prediction]? = nil
                          
             let inputShapedArray =  image.toMLShapedArrayNCWH()
             /*
             for w in 0..<inputShapedArray.shape[2] {
                 print(w,",200"," : ", inputShapedArray[scalarAt: [0,0,w, 200]])
             }
            */
  
             let defaultConfig = MLModelConfiguration()
             let model = try okra_u2net_tensorin(configuration: defaultConfig)
             let results = try model.prediction(x:inputShapedArray)
             
             let mainPredictionResult = results.var_2524ShapedArray
             
             let newPrediction = Prediction(mask: mainPredictionResult)
             if predictions == nil {
                 predictions = [newPrediction]
             } else {
                 predictions!.append(newPrediction)
             }
             
             imagePredictionHandler(predictions, image:image)
             
         } catch {
             print("Vision was unable to make a prediction...\n\n\(error.localizedDescription)")
         }
     }

    /// The method the Image Predictor calls when its image classifier model generates a prediction.
    /// - Parameter predictions: An array of predictions.
    /// - Tag: imagePredictionHandler
    private func imagePredictionHandler(_ predictions: [Prediction]?, image: UIImage) {
        guard let predictions = predictions else {
            updatePredictionLabel("No predictions. (Check console log.)")
            return
        }

        let (isTender, odds, conf) = formatPredictions(predictions)
        showMaskedOkra(predictions, isTender:isTender, image: image)
        
        if (conf > 0.9) {
          let predictionString = "is Okra : (" + String(format: "%.2f", conf * 100) + "% likely) \n" + (isTender ? "Will" : "Will not") + " pass tip-break test : (" + String (format: "%.2f", odds * 100) + "% likely)"
          updatePredictionLabel(predictionString)
        } else {
          let notOkraString = "is not Okra"
          updatePredictionLabel(notOkraString)
        }

        
        
        
    }

    private func drawTestBand(buffer: UnsafeMutablePointer<UInt8>, height: Int, width: Int) -> Void {
        for h in 0..<height {
            for w in 0..<width {
                let linearIndex =  (h * width + w)*3
                buffer[linearIndex] = 255
                buffer[linearIndex + 1] = 0
                buffer[linearIndex + 2] = 0
                
                if (h > 100 && h < 350) {
                    buffer[linearIndex] = 0
                    buffer[linearIndex + 1] = 0
                    buffer[linearIndex + 2] = 255
                }
            }
        }
    }
        
    
    /// Creates an new image view applying predicted Okra segment mask
    private func showMaskedOkra(_ predictions: [Prediction], isTender:Bool, image: UIImage) -> Void {
        let okraSegMask = predictions[0].mask
        let dimensions = okraSegMask.shape
        let width = dimensions[2]
        let height = dimensions[3]
        let data = NSMutableData(length: MemoryLayout<UInt8>.size * 3 * width * height)!
        let buffer = data.mutableBytes.assumingMemoryBound(to: UInt8.self)
        
        // drawTestBand(buffer: buffer, height: height, width: width)
        
        // copy original image colors to newly constructed buffer
        image.copyPerPixelColorToBuffer(buffer: buffer)
        
        // Update buffer to mask out non-Okra backgound pixels
        for h in 0..<height {
            for w in 0..<width {
                let linearIndex =  (h * width + w)*3
                if (okraSegMask[scalarAt: [0,(isTender ? 0:1),w,h]] < Float(0.001)) {
                    buffer[linearIndex + 0] = UInt8(255 * (isTender ? 0:1))
                    buffer[linearIndex + 1] = UInt8(255 * (isTender ? 1:0))
                    buffer[linearIndex + 2] = 0
                }
            }
        }
        
        // Update display
        DispatchQueue.main.async {
            self.imageView.image = self.imageHelper.convertRGBBuffer(toUIImage: buffer , withWidth: Int32(width), withHeight: Int32(height))
        }
        return
    }
    
    /// Converts a prediction's observations into human-readable strings.
    /// - Parameter observations: The classification observations from a Vision request.
    /// - Tag: formatPredictions
    private func formatPredictions(_ predictions: [Prediction]) -> (Bool, Float, Float) {
        // Vision sorts the classifications in descending confidence order.
        var topPredictions = false

        let okraSegMask = predictions[0].mask
        let dimensions = okraSegMask.shape
        // print("okraSegMask type = ", (type(of: okraSegMask)))
        // print("dimensions = ", dimensions)
            
        var goodOkraP50 : Float32 = 0
        var badOkraP50 : Float32 = 0
        var badOkraVote = 0
        var goodOkraVote = 0
        var odds : Float = 0
        var confidence : Float = 0
        
        // print("OkraSegMask type = ", type(of: okraSegMask))

        
        for i in 0..<dimensions[2] {
            for j in 0..<dimensions[3]{
                    
                if okraSegMask[scalarAt: [0,1,i,j]] > 0.5 {
                    badOkraVote += 1
                    badOkraP50 += okraSegMask[scalarAt: [0,1,i,j]]
                }
                    
                if okraSegMask[scalarAt: [0,0,i,j]] > 0.5 {
                    goodOkraVote += 1
                    goodOkraP50 += okraSegMask[scalarAt: [0,0,i,j]]
                }
            }
        }
        
        
        var goodOkraConfidence : Float = 0
        var badOkraConfidence : Float = 0
            
        if (goodOkraVote > 0) {
            goodOkraConfidence = goodOkraP50/Float(goodOkraVote)
        }
        
        if (badOkraVote > 0) {
            badOkraConfidence = badOkraP50/Float(badOkraVote)
        }
        
        /*
        print("good okra P50 = ", goodOkraConfidence, " bad okra P50 = ", badOkraConfidence)
        print("good okra vote = ", goodOkraVote, " bad okra vote = ", badOkraVote)
        */
        
        odds = (badOkraP50/(goodOkraP50 + badOkraP50))
        confidence = badOkraConfidence
        
        if (goodOkraVote > badOkraVote) {
            topPredictions = true
            odds = (goodOkraP50/(goodOkraP50 + badOkraP50))
            confidence = goodOkraConfidence
        }
        
        return (topPredictions, odds, confidence)
    }
}
