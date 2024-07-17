import {
    HandLandmarker,
    FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

const demosSection = document.getElementById("demos");

let handLandmarker = undefined;
let runningMode = "IMAGE";
let enableWebcamButton;
let webcamRunning = false;

// Before we can use HandLandmarker class we must wait for it to finish
// loading. Machine Learning models can be large and take a moment to
// get everything needed to run.
const createHandLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
        },
        runningMode: runningMode,
        numHands: 2
    });
    demosSection.classList.remove("invisible");
};
createHandLandmarker();

/********************************************************************
// Demo 1: Grab a bunch of images from the page and detection them
// upon click.
********************************************************************/

// In this demo, we have put all our clickable images in divs with the
// CSS class 'detectionOnClick'. Lets get all the elements that have
// this class.
const imageContainers = document.getElementsByClassName("detectOnClick");

// Now let's go through all of these and add a click event listener.
for (let i = 0; i < imageContainers.length; i++) {
    // Add event listener to the child element whichis the img element.
    imageContainers[i].children[0].addEventListener("click", handleClick);
}

// When an image is clicked, let's detect it and display results!
async function handleClick(event) {
    if (!handLandmarker) {
        console.log("Wait for handLandmarker to load before clicking!");
        return;
    }

    if (runningMode === "VIDEO") {
        runningMode = "IMAGE";
        await handLandmarker.setOptions({ runningMode: "IMAGE" });
    }
    // Remove all landmarks drawed before
    const allCanvas = event.target.parentNode.getElementsByClassName("canvas");
    for (var i = allCanvas.length - 1; i >= 0; i--) {
        const n = allCanvas[i];
        n.parentNode.removeChild(n);
    }

    // We can call handLandmarker.detect as many times as we like with
    // different image data each time. This returns a promise
    // which we wait to complete and then call a function to
    // print out the results of the prediction.
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

/********************************************************************
// Demo 2: Continuously grab image from webcam stream and detect it.
********************************************************************/

const videoHeight = "640px";
const videoWidth = "480px";

const liveView = document.getElementById("liveView");
const webcamElement = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const viewContainer = document.getElementById("liveView");
const canvasCtx = canvasElement.getContext("2d");

const divToPlace = document.getElementById('ring-to-place-1');

// Check if webcam access is supported.
function hasGetUserMedia() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

// If webcam supported, add event listener to activation button.
if (hasGetUserMedia()) {
    enableWebcamButton = document.getElementById("webcamButton");
    enableWebcamButton.addEventListener("click", enableCam);
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
        // Stop all video streams.
        webcamElement.srcObject.getVideoTracks().forEach((track) => {
            track.stop();
        });
        return;
    }

    runningMode = "VIDEO";
    await handLandmarker.setOptions({ runningMode: "VIDEO" });

    webcamRunning = true;
    enableWebcamButton.innerText = "DISABLE WEBCAM";

    // getUsermedia parameters to force video but not audio.
    const constraints = {
        video: {
            width: { ideal: 4096 },
            height: { ideal: 2160 },
            facingMode: { ideal: "environment" } // Prefer back camera
        }
    };

    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
        webcamElement.srcObject = stream;

        // Check if the facing mode is environment and add flip class
        const videoTracks = stream.getVideoTracks();
        if (videoTracks.length > 0 && videoTracks[0].getSettings().facingMode === 'environment') {
            webcamElement.classList.remove('flip');
            canvasElement.classList.remove('flip');
            canvasElement.classList.add('noflip');
        } else {
            webcamElement.classList.add('flip');
            canvasElement.classList.remove('noflip');
            canvasElement.classList.add('flip');
        }

        webcamElement.addEventListener("loadeddata", predictWebcam);
    }).catch(function (error) {
        console.error("Error accessing the webcam: ", error);
    });
}


async function predictWebcam() {
    canvasElement.style =
        "width: 100%; ";
    /*webcamElement.videoWidth +
    "px;" +
    "height: " +
    webcamElement.videoHeight +
    "px;";*/
    canvasElement.width = webcamElement.videoWidth;
    canvasElement.height = webcamElement.videoHeight;

    const handLandmarkerResult = handLandmarker.detectForVideo(
        webcamElement,
        performance.now()
    );

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    if (handLandmarkerResult.landmarks) {
        for (const landmarks of handLandmarkerResult.landmarks) {
            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
                color: "#00FF00",
                lineWidth: 5
            });
            // drawLandmarks(canvasCtx, landmarks, { color: "#FF0000", lineWidth: 2 });
            drawLandmarksWithNumbers(canvasCtx, landmarks, {
                color: "#FF0000",
                lineWidth: 2
            });

            placeRing(canvasCtx, landmarks[9], landmarks[10]);
        }
    }
    canvasCtx.restore();

    // Call this function again to keep predicting when the browser is ready.
    if (webcamRunning === true) {
        window.requestAnimationFrame(predictWebcam);
    }
}


function drawLandmarksWithNumbers(ctx, landmarks, options) {
    const { color, lineWidth } = options;
    ctx.fillStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.font = "12px Arial";

    for (let i = 0; i < landmarks.length; i++) {
        const landmark = landmarks[i];
        const x = landmark.x * ctx.canvas.width;
        const y = landmark.y * ctx.canvas.height;

        ctx.beginPath();
        ctx.arc(x, y, 5, 0, 2 * Math.PI);
        ctx.fill();

        ctx.fillText(i, x + 5, y + 5); // Display the landmark index number
    }
}

function placeRing(ctx, landmark1, landmark2) {
    // Get canvas dimensions
    const canvasRect = canvasElement.getBoundingClientRect();
    const canvasWidth = canvasRect.width;
    const canvasHeight = canvasRect.height;

    const xL1 = landmark1.x * ctx.canvas.width;
    const yL1 = landmark1.y * ctx.canvas.height;

    const xL2 = landmark2.x * ctx.canvas.width;
    const yL2 = landmark2.y * ctx.canvas.height;

    // Calculate the middle point
    const targetX = (xL1 + xL2) / 2;
    const targetY = (yL1 + yL2) / 2;

    // Calculate rotation angle
    const rotateDeg = Math.atan2(yL2 - yL1, xL2 - xL1) * (180 / Math.PI);

    const containerRect = viewContainer.getBoundingClientRect();
    // Get container dimensions (assuming the container is the same size as the viewport)
    const containerWidth = containerRect.width;
    const containerHeight = containerRect.height;

    // Calculate scale factors
    const scaleX = containerWidth / canvasWidth;
    const scaleY = containerHeight / canvasHeight;

    // Calculate new position in container coordinates
    const newX = targetX * scaleX;
    const newY = targetY * scaleY;

    // Calculate translation values (translateX and translateY)
    const translateX = 0; // Assuming no additional translation needed
    const translateY = 0; // Assuming no additional translation needed

    // Apply styles to the div
    divToPlace.style.position = `absolute`;
    divToPlace.style.left = `${newX}px`;
    divToPlace.style.top = `${newY}px`;
    //divToPlace.style.transform = `rotate(${rotateDeg}deg) translateX(${translateX}px) translateY(${translateY}px)`;
    divToPlace.style.transformOrigin = '0 0';
}