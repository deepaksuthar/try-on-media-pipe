import {
    HandLandmarker,
    FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

import { ImageSegmenter } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2";

class SimpleKalmanFilter {
    constructor(R, Q) {
        this.R = R; // noise covariance
        this.Q = Q; // process covariance
        this.A = 1;
        this.B = 0;
        this.C = 1;
        this.cov = NaN;
        this.x = NaN; // estimated signal without noise
    }

    filter(z) {
        if (isNaN(this.x)) {
            this.x = (1 / this.C) * z;
            this.cov = (1 / this.C) * this.Q * (1 / this.C);
        } else {
            // Compute prediction
            let predX = (this.A * this.x) + (this.B * 0);
            let predCov = ((this.A * this.cov) * this.A) + this.R;

            // Kalman gain
            let K = predCov * this.C * (1 / ((this.C * predCov * this.C) + this.Q));

            // Correction
            this.x = predX + K * (z - (this.C * predX));
            this.cov = predCov - (K * this.C * predCov);
        }

        return this.x;
    }
}

const demosSection = document.getElementById("demos");

let handLandmarker = undefined;
let runningMode = "IMAGE";
let enableWebcamButton;
let captureButton;
let webcamRunning = false;

let imageSegmenter;
let labels;

// Before we can use HandLandmarker class we must wait for it to finish loading.
const createHandLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            //modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`,
            delegate: "GPU"
        },
        runningMode: runningMode,
        numHands: 1,
        minHandDetectionConfidence: 0.7,
        minTrackingConfidence: 0.7
    });
    demosSection.classList.remove("invisible");
};
createHandLandmarker();

async function createImageSegmenter() {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2/wasm");
    imageSegmenter = await ImageSegmenter.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
            delegate: "GPU"
        },
        runningMode: "IMAGE",
        outputCategoryMask: true,
        outputConfidenceMasks: false
    });
}

createImageSegmenter();

const webcamElement = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const viewContainer = document.getElementById("viewContainer");
const canvasCtx = canvasElement.getContext("2d");
const FinalCapturedImage = document.getElementById('CapturedImage')

function hasGetUserMedia() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

if (hasGetUserMedia()) {
    enableWebcamButton = document.getElementById("webcamButton");
    enableWebcamButton.addEventListener("click", enableCam);

    captureButton = document.getElementById("captureButton");
    captureButton.disabled = true; // Initially disable the Capture button
    captureButton.addEventListener("click", captureImage);
} else {
    console.warn("getUserMedia() is not supported by your browser");
}

async function enableCam(event) {
    if (!handLandmarker) {
        console.log("Wait! objectDetector not loaded yet.");
        return;
    }

    if (webcamRunning === true) {
        webcamRunning = false;
        enableWebcamButton.innerText = "ENABLE WEBCAM";
        webcamElement.srcObject.getVideoTracks().forEach((track) => {
            track.stop();
        });
        return;
    }

    runningMode = "VIDEO";
    await handLandmarker.setOptions({ runningMode: "VIDEO" });

    webcamRunning = true;
    enableWebcamButton.style.display = "none";

    const constraints = {
        video: {
            width: { ideal: 1920 },
            height: { ideal: 1080 },
            facingMode: { ideal: "environment" } // Prefer back camera
        }
    };

    navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
        webcamElement.srcObject = stream;

        const videoTracks = stream.getVideoTracks();
        if (videoTracks.length > 0 && videoTracks[0].getSettings().facingMode === 'environment') {
            webcamElement.classList.remove('flip');
            canvasElement.classList.remove('flip');
            canvasElement.classList.add('noflip');
            FinalCapturedImage.classList.remove('flip');
            isFlipped = false;
        } else {
            webcamElement.classList.add('flip');
            canvasElement.classList.add('flip');
            canvasElement.classList.remove('noflip');
            FinalCapturedImage.classList.add('flip');
            isFlipped = true;
        }

        webcamElement.addEventListener("loadeddata", predictWebcam);

        document.getElementById('CapturedImage').style.display = 'none';
        enableWebcamButton.style.display = 'none';
        captureButton.style.display = 'inline-block';

        divToPlace.style.position = `absolute`;
        divToPlace.style.left = `${-100}px`;
        divToPlace.style.top = `${-100}px`;
    });
}

const referencePositions = [
    { "x": 0.48430564999580383, "y": 0.8110854625701904, "z": 5.423733568932221e-7 },
    { "x": 0.5502023696899414, "y": 0.7348806858062744, "z": -0.00948019977658987 },
    { "x": 0.589587926864624, "y": 0.6451688408851624, "z": -0.021786006167531013 },
    { "x": 0.6157299876213074, "y": 0.5460792183876038, "z": -0.032154250890016556 },
    { "x": 0.6464489102363586, "y": 0.4790748953819275, "z": -0.04322601854801178 },
    { "x": 0.5311582684516907, "y": 0.5096843242645264, "z": -0.03053961880505085 },
    { "x": 0.5542333126068115, "y": 0.3711826205253601, "z": -0.04405640438199043 },
    { "x": 0.5643584728240967, "y": 0.2880013585090637, "z": -0.05069946125149727 },
    { "x": 0.5714240670204163, "y": 0.228133887052536, "z": -0.055220432579517365 },
    { "x": 0.48337316513061523, "y": 0.49954313039779663, "z": -0.03383808210492134 },
    { "x": 0.4870733618736267, "y": 0.3366149663925171, "z": -0.047285668551921844 },
    { "x": 0.4889812469482422, "y": 0.24255409836769104, "z": -0.05254475399851799 },
    { "x": 0.4909090995788574, "y": 0.17631328105926514, "z": -0.05623533949255943 },
    { "x": 0.44047462940216064, "y": 0.5183423161506653, "z": -0.03667522221803665 },
    { "x": 0.43307211995124817, "y": 0.36207687854766846, "z": -0.050330400466918945 },
    { "x": 0.43187040090560913, "y": 0.27007052302360535, "z": -0.059464603662490845 },
    { "x": 0.43410253524780273, "y": 0.20186573266983032, "z": -0.06592313200235367 },
    { "x": 0.40458112955093384, "y": 0.5548273324966431, "z": -0.038843441754579544 },
    { "x": 0.38064879179000854, "y": 0.4458256959915161, "z": -0.05402098968625069 },
    { "x": 0.3689345717430115, "y": 0.37663131952285767, "z": -0.06190962344408035 },
    { "x": 0.3627241551876068, "y": 0.31447169184684753, "z": -0.06592313200235367 }
];

function isHandFlat1(landmarks, zThreshold) {
    // Extract relevant landmarks for flatness check
    const landmarksToCheck = [
        landmarks[0],
        landmarks[5],
        landmarks[6],
        landmarks[9],
        landmarks[10],
        landmarks[13],
        landmarks[14]
    ];

    // Define a threshold for acceptable z-coordinate variance
    //const zThreshold = 0.035;

    // Extract z-coordinates from reference landmarks
    const referenceZs = referencePositions.map(l => l.z);

    // Check if the z-coordinates of the hand landmarks are within the threshold of the reference
    return landmarksToCheck.every((landmark, index) => {
        const referenceZ = referenceZs[index];
        return Math.abs(landmark.z - referenceZ) < zThreshold;
    });
}

function isHandFlatWithScreen(landmarks, referenceZ, zThreshold) {
    // Landmarks to check for flatness
    const indicesToCheck = [5, 6, 9, 10, 13, 14];

    indicesToCheck.forEach(index => {
        const landmark = landmarks[index];
        if (landmark) {
            const difference = Math.abs(landmark.z - referenceZ);
            console.log(`Landmark ${index}:`);
            console.log(`Math.abs(landmark.z - referenceZ): ${difference}`);
            console.log(`zThreshold: ${zThreshold}`);
            console.log(`Difference: ${difference > zThreshold ? 'Out of threshold' : 'Within threshold'}`);
        } else {
            console.log(`Landmark ${index} not found.`);
        }
    });
}


async function predictWebcam() {
    canvasElement.width = webcamElement.videoWidth;
    canvasElement.height = webcamElement.videoHeight;

    const minSide = Math.min(canvasElement.width, canvasElement.height);
    const squareSize = minSide * 0.9;
    const squareX = (canvasElement.width - squareSize) / 2;
    const squareY = (canvasElement.height - squareSize) / 2;

    // Now let's start detecting the stream.
    if (webcamRunning === true) {
        const startTimeMs = performance.now();

        const handLandmarkerResult = await handLandmarker.detectForVideo(
            webcamElement,
            startTimeMs
        );

        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        canvasCtx.drawImage(
            webcamElement,
            0,
            0,
            canvasElement.width,
            canvasElement.height
        );

        // Draw the square
        canvasCtx.strokeStyle = "#FFFF00";
        canvasCtx.lineWidth = 4;
        canvasCtx.strokeRect(squareX, squareY, squareSize, squareSize);

        let handDetected = false;
        let allLandmarksInSquare = false;
        let handFlat = false;

        if (handLandmarkerResult.landmarks) {
            for (const landmarks of handLandmarkerResult.landmarks) {
                drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
                    color: "#00FF00",
                    lineWidth: 5
                });
                drawLandmarks(canvasCtx, landmarks, {
                    color: "#FF0000",
                    lineWidth: 2
                });

                handDetected = true;

                // Check if all landmarks are inside the square
                allLandmarksInSquare = landmarks.every(landmark => {
                    const x = landmark.x * canvasElement.width;
                    const y = landmark.y * canvasElement.height;
                    return (
                        x >= squareX &&
                        x <= squareX + squareSize &&
                        y >= squareY &&
                        y <= squareY + squareSize
                    );
                });

                // Check if the hand is flat
                handFlat = isHandFlat1(landmarks, 0.08 * squareSize / 640);

                if (allLandmarksInSquare && handFlat) {
                    //console.log("Hand is in the square!");
                    landmarksToSave = landmarks;
                }

            }


        }

        canvasCtx.restore();

        // Enable capture button only when hand is detected, all landmarks are in the square, and the hand is flat
        captureButton.disabled = !(handDetected && allLandmarksInSquare && handFlat);

        // Call this function again to keep predicting when the browser is ready
        window.requestAnimationFrame(predictWebcam);
    }
}

var landmarksToSave;

function captureImage() {
    // Create a new canvas to capture the current frame
    const canvas = document.createElement('canvas');
    canvas.width = canvasElement.width;
    canvas.height = canvasElement.height;
    const context = canvas.getContext('2d');

    // Draw the current frame from the webcam onto the canvas
    context.drawImage(webcamElement, 0, 0, canvas.width, canvas.height);

    //placeRing(context, landmarksToSave);

    // Convert the canvas content to a data URL
    const capturedImage = canvas.toDataURL('image/png');

    // Display the captured image
    document.getElementById('CapturedImage').src = capturedImage;
    document.getElementById('CapturedImage').style.display = 'block';

    // Stop the webcam
    if (webcamElement.srcObject) {
        webcamElement.srcObject.getTracks().forEach(track => track.stop());
    }

    // Show the enable camera button and hide the capture button
    enableWebcamButton.style.display = 'inline-block';
    captureButton.style.display = 'none';

    // Reset webcam running state
    webcamRunning = false;


    processCapturedImage(capturedImage);
}


const processCapturedImage = async (capturedImage) => {

    const img = new Image();
    img.src = capturedImage;

    // Wait for the image to load, then pass it to handLandmarker.detect
    img.onload = async () => {


        const canvas = document.createElement('canvas');
        canvas.width = canvasElement.width;
        canvas.height = canvasElement.height;
        const cxt = canvas.getContext('2d');


        runningMode = "IMAGE";
        await handLandmarker.setOptions({ runningMode: "IMAGE" });

        // Detect hand landmarks
        const handLandmarkerResult = await handLandmarker.detect(img);

        if (handLandmarkerResult.landmarks.length > 0) {
            const landmarks = handLandmarkerResult.landmarks[0];

            // Create body-skin mask
            const result = await imageSegmenter.segment(img);
            labels = imageSegmenter.getLabels();
            const imageData = cxt.getImageData(0, 0, canvas.width, canvas.height);
            const mask = result.categoryMask.getAsUint8Array();

            for (let i = 0; i < mask.length; i++) {
                const category = labels[mask[i]];
                if (category === "body-skin") {
                    const alpha = mask[i] ? 255 : 0;
                    imageData.data[i * 4 + 0] = 0;
                    imageData.data[i * 4 + 1] = 0;
                    imageData.data[i * 4 + 2] = 0;
                    imageData.data[i * 4 + 3] = alpha;
                } else {
                    imageData.data[i * 4 + 3] = 0;
                }
            }

            cxt.putImageData(imageData, 0, 0);



            // Calculate width of middle finger part between landmarks 9 and 10
            const canvasWidth = canvas.width;
            const canvasHeight = canvas.height;

            const xL1 = landmarks[9].x * canvasWidth;
            const yL1 = landmarks[9].y * canvasHeight;
            const xL2 = landmarks[10].x * canvasWidth;
            const yL2 = landmarks[10].y * canvasHeight;

            // Calculate the midpoint between points 9 and 10
            const midX = xL2;//(xL1 + xL2) / 2;
            const midY = yL2;//(yL1 + yL2) / 2;

            // Calculate the direction vector perpendicular to the line connecting points 9 and 10
            const deltaX = xL2 - xL1;
            const deltaY = yL2 - yL1;
            const perpendicularX = -deltaY;
            const perpendicularY = deltaX;

            // Normalize the perpendicular vector
            const magnitude = Math.sqrt(perpendicularX * perpendicularX + perpendicularY * perpendicularY);
            const unitX = perpendicularX / magnitude;
            const unitY = perpendicularY / magnitude;

            // Find the width by checking in the perpendicular direction
            let leftDistance = 0;
            let rightDistance = 0;

            let lPoint, rPoint;

            // Move in the negative direction (left side)
            while (true) {
                const checkX = Math.round(midX + leftDistance * unitX);
                const checkY = Math.round(midY + leftDistance * unitY);
                const alphaIndex = (checkY * canvasWidth + checkX) * 4 + 3; // alpha is at the 4th position in RGBA

                if (checkX < 0 || checkX >= canvasWidth || checkY < 0 || checkY >= canvasHeight || imageData.data[alphaIndex] === 0) {
                    lPoint = { x: checkX, y: checkY };
                    break;
                }
                leftDistance--;
            }

            // Move in the positive direction (right side)
            while (true) {
                const checkX = Math.round(midX + rightDistance * unitX);
                const checkY = Math.round(midY + rightDistance * unitY);
                const alphaIndex = (checkY * canvasWidth + checkX) * 4 + 3; // alpha is at the 4th position in RGBA

                if (checkX < 0 || checkX >= canvasWidth || checkY < 0 || checkY >= canvasHeight || imageData.data[alphaIndex] === 0) {
                    rPoint = { x: checkX, y: checkY };
                    break;
                }
                rightDistance++;
            }

            const middleFingerWidth = Math.abs(leftDistance) + rightDistance;
            const cPoint = { x: (lPoint.x + rPoint.x) / 2, y: (lPoint.y + rPoint.y) / 2 };
            console.log('Segmentation Completed. Middle Finger Width:' + middleFingerWidth.toFixed(2));
            console.log('Segmentation Completed. Middle Finger point:' + cPoint.x + " : " + cPoint.y);

            // Draw the point at cPoint
            cxt.beginPath();
            cxt.arc(cPoint.x, cPoint.y, 2, 0, 2 * Math.PI); // Draws a circle with radius 2
            cxt.fillStyle = 'red'; // Set the color of the point
            cxt.fill(); // Fill the circle with the set color

            // Optionally, draw the left and right points as well for visual reference
            cxt.beginPath();
            cxt.arc(lPoint.x, lPoint.y, 2, 0, 2 * Math.PI); // Draw left point
            cxt.fillStyle = 'blue'; // Set the color of the left point
            cxt.fill();

            cxt.beginPath();
            cxt.arc(rPoint.x, rPoint.y, 2, 0, 2 * Math.PI); // Draw right point
            cxt.fillStyle = 'green'; // Set the color of the right point
            cxt.fill();

            document.getElementById("CapturedImageSeg").src = canvas.toDataURL('image/png');;
            placeRing(cxt, landmarks);

        } else {
            console.log('No hand detected');
        }
    };
};


const divToPlace = document.getElementById('ring-to-place-1');
const middleRing1 = document.getElementById('ring-1');

let isFlipped = false;
function placeRing(ctx, landmarks) {

    const landmark1 = landmarks[9];
    const landmark2 = landmarks[10];

    // Get canvas dimensions
    const canvasRect = canvasElement.getBoundingClientRect();
    const canvasWidth = canvasRect.width;
    const canvasHeight = canvasRect.height;

    const xL1 = landmark1.x * canvasWidth;
    const yL1 = landmark1.y * canvasHeight;

    const xL2 = landmark2.x * canvasWidth;
    const yL2 = landmark2.y * canvasHeight;

    const targetPosition = getRingPlacementAt(landmark1, landmark2);

    // Calculate the middle point
    /*const targetX = (xL1 + xL2) / 2;
    const targetY = (yL1 + yL2) / 2;*/

    const targetX = targetPosition.x * canvasWidth;
    const targetY = targetPosition.y * canvasHeight;

    // Calculate rotation angle
    let rotateDeg = Math.atan2(yL2 - yL1, xL2 - xL1) * (180 / Math.PI);

    const containerRect = viewContainer.getBoundingClientRect();
    // Get container dimensions (assuming the container is the same size as the viewport)
    const containerWidth = containerRect.width;
    const containerHeight = containerRect.height;

    // Calculate scale factors
    const scaleX = containerWidth / canvasWidth;
    const scaleY = containerHeight / canvasHeight;

    console.log(scaleX + " : " + scaleY);
    console.log(containerWidth + " : " + canvasWidth);
    console.log(containerHeight + " : " + canvasHeight);

    // Calculate new position in container coordinates
    let newX = targetX;// * scaleX;
    const newY = targetY;// * scaleY;

    //let newX = kalmanFilterX.filter(targetX * scaleX);
    //const newY = kalmanFilterY.filter(targetY * scaleY);

    // Calculate translation values (translateX and translateY)
    const translateX = 0; // Assuming no additional translation needed
    const translateY = 0; // Assuming no additional translation needed

    const landmarkW1 = landmarks[5];
    const landmarkW2 = landmarks[9];

    const xLW1 = landmarkW1.x * canvasWidth;
    const yLW1 = landmarkW1.y * canvasHeight;

    const xLW2 = landmarkW2.x * canvasWidth;
    const yLW2 = landmarkW2.y * canvasHeight;

    let fingerWidth = Math.sqrt(Math.pow(xLW2 - xLW1, 2) + Math.pow(yLW2 - yLW1, 2));

    console.log(`Finger width (without z): ${fingerWidth}`);
    fingerWidth = Calculate3DWidth(landmarkW1, landmarkW2, canvasWidth, canvasHeight);

    divToPlace.style.width = `${fingerWidth * 0.95}px`;
    divToPlace.style.height = `${fingerWidth * 0.95}px`;
    // Apply styles to the div
    divToPlace.style.position = `absolute`;

    newX = isFlipped ? (canvasWidth - newX) : newX;
    //rotateDeg+= 90;

    rotateDeg = isFlipped ? -rotateDeg : rotateDeg;

    divToPlace.style.left = `${newX}px`;
    divToPlace.style.top = `${newY}px`;
    divToPlace.style.transform = `rotate(${rotateDeg + 90}deg) translateX(${translateX}px) translateY(${translateY}px)`;
    divToPlace.style.transformOrigin = '0 0';

    console.log();
    checkHandSide(landmarks);
}

function Calculate3DWidth(landmarkW1, landmarkW2, canvasWidth, canvasHeight) {

    const zCompo = (canvasWidth + canvasHeight);// / 2;

    const xLW1 = landmarkW1.x * canvasWidth;
    const yLW1 = landmarkW1.y * canvasHeight;
    const zLW1 = landmarkW1.z * zCompo; // Z coordinate of landmark 5

    const xLW2 = landmarkW2.x * canvasWidth;
    const yLW2 = landmarkW2.y * canvasHeight;
    const zLW2 = landmarkW2.z * zCompo; // Z coordinate of landmark 9

    // Calculate the finger width considering the z-coordinate as well
    const fingerWidth = Math.sqrt(
        Math.pow(xLW2 - xLW1, 2) +
        Math.pow(yLW2 - yLW1, 2) +
        Math.pow(zLW2 - zLW1, 2)  // Incorporating the z-coordinate
    );

    console.log(`Finger width (with z): ${fingerWidth}`);

    return fingerWidth;
}


function getRingPlacementAt(x, y) {

    // Calculate the vector from x to y
    const vectorXY = {
        x: y.x - x.x,
        y: y.y - x.y
    };

    // Calculate the length of the vector
    const lengthXY = Math.sqrt(vectorXY.x ** 2 + vectorXY.y ** 2);
    var distance = lengthXY * 3 / 5;
    const d = distance;
    // Normalize the vector (unit vector in the direction of y from x)
    const unitVectorXY = {
        x: vectorXY.x / lengthXY,
        y: vectorXY.y / lengthXY
    };

    // Scale the unit vector by the desired distance d
    const scaledVector = {
        x: unitVectorXY.x * d,
        y: unitVectorXY.y * d
    };

    // Find the new point at distance d from x
    const newPoint = {
        x: x.x + scaledVector.x,
        y: x.y + scaledVector.y
    };

    console.log("New point at distance d from x:", newPoint);
    return newPoint;
}

function checkHandSide(landmarks) {
    // Example coordinates (replace with actual coordinates)
    const wrist = landmarks[0]
    const thumb = landmarks[1]
    const pinky = landmarks[17]

    // Calculate vectors
    const vectorWristToThumb = {
        x: thumb.x - wrist.x,
        y: thumb.y - wrist.y,
        z: thumb.z - wrist.z
    };

    const vectorWristToPinky = {
        x: pinky.x - wrist.x,
        y: pinky.y - wrist.y,
        z: pinky.z - wrist.z
    };

    // Compute cross product
    const crossProduct = {
        x: vectorWristToThumb.y * vectorWristToPinky.z - vectorWristToThumb.z * vectorWristToPinky.y,
        y: vectorWristToThumb.z * vectorWristToPinky.x - vectorWristToThumb.x * vectorWristToPinky.z,
        z: vectorWristToThumb.x * vectorWristToPinky.y - vectorWristToThumb.y * vectorWristToPinky.x
    };

    // Determine hand orientation
    if (crossProduct.z > 0) {
        console.log("Right hand");
        middleRing1.style.left = `${-56}%`;
    } else {
        console.log("Left hand");
        middleRing1.style.left = `${-42}%`;
    }

}