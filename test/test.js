const imageUpload = document.getElementById('imageUpload');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');

// Load and initialize the ImageSegmenter
let imageSegmenter;

import {
    ImageSegmenter
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";


(async () => {
  imageSegmenter = await ImageSegmenter.createFromOptions(Vision.ImageSegmenter.ImageSegmenterOptions.builder()
    .setModelAssetPath('https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite')
    .setRunningMode(RunningMode.IMAGE)
    .build());


imageSegmenter.onResults((results) => {
  canvasElement.width = results.width;
  canvasElement.height = results.height;

  // Get the segmentation mask for the skin class (assuming class index 1 is skin)
  const skinMask = results.segmentedMasks[0].map((value, index) => {
    return (value === 1) ? 255 : 0;  // Assuming 1 is the class ID for skin
  });

  const imageData = canvasCtx.createImageData(canvasElement.width, canvasElement.height);
  for (let i = 0; i < skinMask.length; i++) {
    const offset = i * 4;
    imageData.data[offset] = results.imageData.data[offset];
    imageData.data[offset + 1] = results.imageData.data[offset + 1];
    imageData.data[offset + 2] = results.imageData.data[offset + 2];
    imageData.data[offset + 3] = skinMask[i];  // Set the alpha channel
  }

  // Draw the segmented skin region with alpha mask
  canvasCtx.putImageData(imageData, 0, 0);
});

imageUpload.addEventListener('change', async (event) => {
  const file = event.target.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = async () => {
        canvasElement.width = img.width;
        canvasElement.height = img.height;
        canvasCtx.drawImage(img, 0, 0, img.width, img.height);

        // Send the image to the MediaPipe Image Segmenter
        await imageSegmenter.segment(img);
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }
});
})();