import {
    HandLandmarker,
    FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

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

// Demo 1: Grab a bunch of images from the page and detect them upon click.
const imageContainers = document.getElementsByClassName("detectOnClick");

for (let i = 0; i < imageContainers.length; i++) {
    imageContainers[i].children[0].addEventListener("click", handleClick);
}

async function handleClick(event) {
    if (!handLandmarker) {
        console.log("Wait for handLandmarker to load before clicking!");
        return;
    }

    if (runningMode === "VIDEO") {
        runningMode = "IMAGE";
        await handLandmarker.setOptions({ runningMode: "IMAGE" });
    }

    const allCanvas = event.target.parentNode.getElementsByClassName("canvas");
    for (var i = allCanvas.length - 1; i >= 0; i--) {
        const n = allCanvas[i];
        n.parentNode.removeChild(n);
    }

    const handLandmarkerResult = await handLandmarker.detect(event.target);
    const canvas = document.createElement("canvas");
    canvas.setAttribute("class", "canvas");
    canvas.setAttribute("width", event.target.naturalWidth + "px");
    canvas.setAttribute("height", event.target.naturalHeight + "px");
    canvas.style =
        "left: 0px;" +
        "top: 0px;" +
        "width: " +
        event.target.width +
        "px;" +
        "height: " +
        event.target.height +
        "px;";

    event.target.parentNode.appendChild(canvas);
    const cxt = canvas.getContext("2d");

    for (const landmarks of handLandmarkerResult.landmarks) {
        drawConnectors(cxt, landmarks, HAND_CONNECTIONS, {
            color: "#00FF00",
            lineWidth: 5
        });
        drawLandmarks(cxt, landmarks, { color: "#FF0000", lineWidth: 2 });
    }
}

// Demo 2: Continuously grab image from webcam stream and detect it.
const videoHeight = "640px";
const videoWidth = "480px";

const liveView = document.getElementById("liveView");
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
                handFlat = isHandFlat1(landmarks, 0.05);// * squareSize / 640);

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

    placeRing(context, landmarksToSave);

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
}


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

    const fingerWidth = Math.sqrt(Math.pow(xLW2 - xLW1, 2) + Math.pow(yLW2 - yLW1, 2));

    divToPlace.style.width = `${fingerWidth * 0.9}px`;
    divToPlace.style.height = `${fingerWidth * 0.9}px`;
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


function getRingPlacementAt(x,y) {
   
    // Calculate the vector from x to y
    const vectorXY = {
        x: y.x - x.x,
        y: y.y - x.y
    };

    // Calculate the length of the vector
    const lengthXY = Math.sqrt(vectorXY.x ** 2 + vectorXY.y ** 2);
    var distance = lengthXY * 3/5;
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