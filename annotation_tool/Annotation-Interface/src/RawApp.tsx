import styled from "styled-components"
import { IStyled } from "./type"
import { Scene } from "./component/Scene"
import React, { useContext, useEffect, useState, useRef, ChangeEvent } from "react"
import { ModelLoader } from "./component/ModelLoader"
import { SwitchModeButton, ProcessCoordinatesButton, OtherButton_Red, OtherButton_Green } from "./component/Button"
import AuthContext from './AuthContext';
import ProgressBar from './component/ProgressBar';
import StyledImageSlider from './component/ImageSlider'

const host = process.env.REACT_APP_API_HOST || "localhost";
const port = process.env.REACT_APP_API_PORT || 8000;
const TAPIR_host = process.env.REACT_APP_TAPIR_HOST || "localhost";
const TAPIR_port = process.env.REACT_APP_TAPIR_PORT || 8000;

const AUTO_PROCESS_COORDINATE = false
const POINTS_TRACKING = false


// Define the interface for the coordinates
interface Coordinate {
	x: number;
	y: number;
  }

// 
  

interface ModalProps {
	isOpen: boolean;
	onClose: () => void;
	children: React.ReactNode;
  }

  const Modal: React.FC<ModalProps> = ({ isOpen, onClose, children }) => {
	if (!isOpen) return null;
  
	return (
	  <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0, 0, 0, 0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
		<div style={{ background: 'white', padding: 20, borderRadius: 5 }}>
		  {children}
		  <button onClick={onClose} style={{ marginTop: 10 }}>Close</button>
		</div>
	  </div>
	);
  };


  interface AppProps extends IStyled {}
  
const RawApp = (props: AppProps) => {


	////////////////////// Code for a select  box to share annotations /////////////////////////

	const [isModalOpen, setIsModalOpen] = useState(false);
  	const [selectedShareFrames, setSelectedShareFrames] = useState<string[]>([]);
	  const handleShareAnnotation = async () => {
		setIsModalOpen(true); // Open the modal
	  };

	const handleSelectChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
		console.log("current annotating frame: ", selectedFrame);
		// Use Array.from to convert options to a real array
		// Filter for options that are selected and map to their value
		const selectedOptions = Array.from(event.target.options)
									 .filter(option => option.selected)
									 .map(option => option.value);
		setSelectedShareFrames(selectedOptions);
		console.log("selectedShareFrames: ", selectedOptions);
		
	  };
	// useEffect(() => {
	// 	console.log("selectedShareFrames: ", selectedShareFrames);
	// }, [selectedShareFrames]);

	const confirmSelection = () => {
		console.log('Selected Frames:', selectedShareFrames);
		// Ask the user to confirm their selection
		if (selectedShareFrames.length === 0) {
		  alert('Please select at least one frame! Or click close to cancel share annotations');
		  return;
		}
		const r = window.confirm("Share with frame " + selectedShareFrames + "?")
		if (!r){
			setIsModalOpen(false); // Close the modal
			setSelectedShareFrames([]);
		}
		else{
			handleSaveProjection(true); //Use share frames, so set to true
		}


		// Add your logic here to handle the confirmed selection,
		// such as processing data or closing the modal.
		closeModal(); // Optionally close the modal after confirmation
	  };
	  
	
	const closeModal = () => {
	setIsModalOpen(false); // Close the modal
	};


	/////////////////////End of Code for a select  box to share annotations /////////////////////////

	// user name
	const { user } = useContext(AuthContext);
	// Add a state variable to keep track of the returned image
	const [returnedProjectImagePaths, setReturnedProjectImagePaths] = useState<string[]>([]);
	// Add a state variable to keep track of the returned image
	const [returnedSegmentationImagePaths, setReturnedSegmentationImagePaths] = useState<string[]>([]);

	// Set up a state variable for the button toggled state
	const [Projection, setProjection] = useState(false); // Two mode: Projection and Segmentation
	const [coordinates3D, setCoordinates3D] = useState<THREE.Vector3[]>([]);
	const [mode, setMode] = useState('Add'); // 'Add' or 'Remove'
	const [positiveKeypoints, setPositiveKeypoints] = useState<Coordinate[]>([]);
	const [negativeKeypoints, setNegativeKeypoints] = useState<Coordinate[]>([]);
	const [confidences, setConfidences] = useState<number[]>([]);
	const [originalImagePath, setOriginalImagePath] = useState('');

	const [currentVideoPath, setCurrentVideoPath] = useState(originalImagePath);
	const [currentUpperImage, setCurrentUpperImage] = useState(originalImagePath);
	

	// the image for current annoation (get from backend by using currentTime)
	const [currentAnnotationImage, setCurrentAnnotationImage] = useState(originalImagePath);
	// Change from frame to video 
	const [selectedVideo, setSelectedVideo] = useState('');
	const [videos, setVideos] = useState<string[]>([]);
	const [currentFrameTime, setCurrentFrameTime] = useState(0);
	const [nextFrameTime, setNextFrameTime] = useState(0);

	const [Categories, setCategories] = useState([]);
	const [selectedCategory, setSelectedCategory] = useState('');
	const [selectedSubCategory, setSelectedSubCategory] = useState('');
	const [subCategories, setSubCategories] = useState([]);
	const [selectedStep, setSelectedStep] = useState('');
	const [steps, setSteps] = useState([]);
	
	//model file paths is a list of several list of strings, each list of strings is the file paths for one step
	const [modelFilePathLists, setModelFilePathLists] = useState<string[][]>([]);
	const [allModelFilePaths, setAllModelFilePaths] = useState<string[]>([]);
	const [currentModelFilePaths, setCurrentModelFilePaths] = useState<string[]>([]);
	const[selectedProjctionImage, setSelectedProjectionImage] = useState('');
	const [isClean3DClicked, setIsClean3DClicked] = useState(false);
	const [CoordLs2D, setCoordLs2D] = useState<Coordinate[]>([]); // The list of 2D coordinates
	const [ManualImagePath, setManualImagePath] = useState(''); // The manual image for current step
	const [input1, setInput1] = useState(''); // For the first input box
	const [input2, setInput2] = useState(''); // For the second input box
	const [input3, setInput3] = useState(''); // For the third input box
	const [selectedSegmentationImage, setSelectedSegmentationImage] = useState(''); // The image to be annotated
	const [messages, setMessages] = useState([]); // Messages to be displayed in the message box
	const imgRef = useRef<HTMLImageElement>(null); // Reference to the image element

	//Image helper
	const [showImageHelper, setShowImageHelper] = useState(false); // Show image helper or not
	const [canvasImageSrc, setCanvasImageSrc] = useState(''); // The image to be annotated

	// Variables for video frame reannotation
	const [currentVideo, setCurrentVideo] = useState(''); // The video to be annotated, returned from backend
	const videoRef = useRef<HTMLVideoElement>(null);
	const canvasRef = useRef<HTMLCanvasElement>(null);
	const [selectedPointIndex, setSelectedPointIndex] = useState<number | null>(null); //for points moving on canvas
	// const [pointsCoordinates, setPointsCoordinates] = useState<Coordinate[]>([]);
	const [canvasImage, setCanvasImage] = useState<HTMLImageElement | null>(null);

	const [isDragging, setIsDragging] = useState(false);

	//Disable mouse click on the 2d image when processing coordinates
	const [disableImageClick, setDisableImageClick] = useState(false);

	const [pointsInitilized, setPointsInitilized] = useState(false);

	const [highlightColors, setHighlightColors] = useState<Record<string, string>>({});
	const [annotationForCurrentFrameExist, setAnnotationForCurrentFrameExist] = useState(false);
	const [hasNewAnnotation, setHasNewAnnotation] = useState(false);


	// States for progress bar:
	const [totalCurrPartCurrStep, setTotalCurrPartCurrStep] = useState(0);
	const [annotatedCurrPartCurrStep, setAnnotatedCurrPartCurrStep] = useState(0);
	const [totalCurrStepCurrVideo, setTotalCurrStepCurrVideo] = useState(0);
	const [annotatedCurrStepCurrVideo, setAnnotatedCurrStepCurrVideo] = useState(0);
	const [totalCurrStep, setTotalCurrStep] = useState(0);
	const [annotatedCurrStep, setAnnotatedCurrStep] = useState(0);
	const [totalCurrVideo, setTotalCurrVideo] = useState(0);
	const [annotatedCurrVideo, setAnnotatedCurrVideo] = useState(0);
	const [totalCurrFurniture, setTotalCurrFurniture] = useState(0);
	const [annotatedCurrFurniture, setAnnotatedCurrFurniture] = useState(0);
	const [totalAllFurniture, setTotalAllFurniture] = useState(0);
	const [annotatedAllFurniture, setAnnotatedAllFurniture] = useState(0);
	const [frameIssueMessage, setFrameIssueMessage] = useState('');

	//Image Helper 2, is used for control whether show the full 3d part or only the part that are annotating.
	const [showImageHelper2, setShowImageHelper2] = useState(false); // Show image helper or not

	const [images, setImages] = useState<string[]>([]); // The list of images for the image slider

    // Handler for the image slider
    const handleSliderChange = (event: ChangeEvent<HTMLInputElement>) => {
        setCurrentUpperImage(images[Number(event.target.value)]);
		console.log("images[Number(event.target.value)]: ", images[Number(event.target.value)]);
    };
	const [PrevMaskImagePath, setPrevMaskImagePath] = useState('');
	const [PrevMaskImageOverlayPath, setPrevMaskImageOverlayPath] = useState('');
	const [PrevMaskReason, setPrevMaskReason] = useState('');
	

	



	
	////////////////Message to the frame btn ////////////////////////
	const handleReportMissingComponent = () => {
		updateProgressBar()
		handleClean()
		const videoElement = videoRef.current;
		if (!videoElement) return;
		videoElement.pause();

		const fps = 1;
		const frameDuration = 1 / fps;

		// Set the video element's current time to the next frame time
		videoElement.currentTime = currentFrameTime + frameDuration; // Move to the next frame
		setNextFrameTime(prevTime => prevTime + frameDuration); //Move to the next frame


		// Add a one-time event listener for 'seeked'
		videoElement.addEventListener('seeked', async () => {
		const canvasElement = canvasRef.current;


		if (!videoElement || !canvasElement) {
			console.error("Video or canvas element is not available");
			return;
		}

		const ctx = canvasElement.getContext("2d");
		if (!ctx) {
			console.error("Unable to get 2D context from canvas");
			return;
		}

		setNextFrameTime(prevTime => prevTime + 1); //Move to the next frame

		// Get the image from backend and draw it on the canvas
		//console.log('start trying to come back to the previous anno')
        try {
			setHasNewAnnotation(false)
            const response = await fetch(`http://${host}:${port}/part-not-able-to-present`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    "user": user,
                    "videoPath": selectedVideo,
                    "Category": selectedCategory,
                    "SubCategory": selectedSubCategory,
                    "Step": selectedStep,
                    'currentModelFilePaths': currentModelFilePaths,
                    'nextFrameTime': nextFrameTime,
					'Projection': Projection,
					'NotAbleToAnnotate': true,
					'Message': frameIssueMessage,
					'ReportMessage': true,
                })
            });

            const data = await response.json();
			if (data.messages){
				setMessages(data.messages);
			}
			if (data.error){
				//console.log("Error received:", data);
				alert(data.error);
				if (data.error === "You have reached the end of the video."){
					setNextFrameTime(0);
					setCurrentFrameTime(0);
					setAnnotationForCurrentFrameExist(false)
					
					return;
				}
			}
			if(data.imagePath != null){
				setAnnotationForCurrentFrameExist(true)
				const unique_suffix = "?t=" + new Date().getTime();
				setCurrentUpperImage(data.imagePath + unique_suffix);
			}else{
				setAnnotationForCurrentFrameExist(false)
				setCurrentUpperImage('');
				
			}
		
			//Set current frame time and set video element's current time
			setCurrentFrameTime(data.currentFrameTime);
			videoElement.currentTime = data.currentFrameTime;

			if (data.base64Image != null){
				alert('Issue reported, moved to the next frame! Success!')
				updateProgressBar()
			}
			else{
				alert('Moved to the next failed!')
			}
			//console.log("Data received:", data);
			const base64Image = data.base64Image;

			// Reshape the image to fit the canvas
			const image = new Image();
			image.src = 'data:image/png;base64,' +base64Image;
			image.onerror = async function(error) {
				console.error("Error loading the image:", error);
			};
			
			image.onload = () => {
				// Set canvas dimensions to match the video
				const container = document.getElementById("middle-part");
				if (container) {
					const containerWidth = container.clientWidth;
					const containerHeight = container.clientHeight;

					// Make sure the image still in the original aspect ratio
					const scale = Math.min(containerWidth / image.width, containerHeight / image.height);
					const scaledWidth = image.width * scale;
					const scaledHeight = image.height * scale;
					const left = (containerWidth - scaledWidth) / 2;
					const top = (containerHeight - scaledHeight) / 2;
					//console.log(`left, top: (${left}, ${top})`);
					//console.log(`scaledWidth, scaledHeight: (${scaledWidth}, ${scaledHeight})`);

					canvasElement.width = scaledWidth;
					canvasElement.height = scaledHeight;
					//console.log(`containerWidth, containerHeight: (${containerWidth}, ${containerHeight})`);
				}

				// Clean the canvas first
				if (ctx) {
					ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
				}
				// Draw the current video frame on the canvas
				ctx.drawImage(image, 0, 0, canvasElement.width, canvasElement.height);
				//console.log("image.src: ", image.src);
				setCanvasImage(image);
				//console.log("image: ", image);
			};
		
		
        } catch (error) {
			if (error instanceof Error) {
				if (error.message.includes('ERR_CONNECTION_REFUSED') || error.message.includes('Network request failed') || error.message.includes('Failed to fetch')) {
					// Here you can handle the specific error
					// For example, by refreshing the page:
					console.log('Connection refused. Refreshing page...');
					window.location.reload();
				}
			}
            console.error('Error:', error);
        }


		const canvasImage = new Image();
		canvasImage.src = canvasElement.toDataURL("image/png");
		const canvasURL = canvasElement.toDataURL("image/png")
		//console.log("canvasURL: ", canvasURL);
		setCanvasImage(canvasImage);
		//console.log("canvasImage: ", canvasImage);
		//console.log("videoElement: ", videoElement);
	
		
		canvasElement.style.display = 'block';
	}  , { once: true });
		
	}
	const handleFrameIssueMessage = (e: React.ChangeEvent<HTMLInputElement>) => {
		setFrameIssueMessage(e.target.value);
	}

	//Color for points
	const pointColors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'brown',  'lavender', 'teal', 'tomato', 'gold', 'silver', 'olive', 'maroon', 'navy', 'lime', 'aqua', 'fuchsia', 'black', 'white'];
	
	// useEffect (() => {
	// 	//console.log("currentUpperImage")
	// 	//console.log(currentUpperImage)
	// }
	// , [currentUpperImage])

	//////////// async functions to move points on the canvas//////////////
	// async function for re-draw the video frame onto canvas
	const drawVideoFrame = () => {
		//console.log('drawVideoFrame')
		const canvasElement = canvasRef.current;
		const ctx = canvasElement?.getContext("2d");
		if (!canvasElement || !ctx) return;
	  
		if (canvasImage) {
		  ctx.drawImage(canvasImage, 0, 0, canvasElement.width, canvasElement.height);
		}
	  };
	  
	  useEffect(() => {
		drawVideoFrame();  // Draw the image first
		drawPoints();      // Then draw the points
	  }, [positiveKeypoints, CoordLs2D, negativeKeypoints,canvasImage]);  // Do this whenever points or canvasImage changes


	  const drawPoints = () => {

		const canvasElement = canvasRef.current;
		if (!canvasElement) return;
		const ctx = canvasElement.getContext("2d");
		if (!ctx) return;

	  
		// Clear
		ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
		drawVideoFrame();

		// Draw Brush Strokes
			console.log("drawPoints called for brush mode");
			console.log("PositiveBrushStrokes.length: ", PositiveBrushStrokes.length);
			console.log("NegativeBrushStrokes.length: ", NegativeBrushStrokes.length);
			// Draw brush strokes
			if (PositiveBrushStrokes.length>0){
				console.log('Drawing PositiveBrushStrokes')
				ctx.strokeStyle = 'skyblue'
				PositiveBrushStrokes.forEach(stroke => {
				ctx.beginPath();
				stroke.forEach((point, index) => {
					ctx.lineWidth = point.brushSize/2.7;
					if (index === 0) {
					ctx.moveTo(point.x * canvasElement.width, point.y * canvasElement.height);
					} else {
					ctx.lineTo(point.x * canvasElement.width, point.y * canvasElement.height);
					}
				});
				ctx.stroke();
				});
			}
			if (NegativeBrushStrokes.length>0){
				console.log('Drawing NegativeBrushStrokes')
				ctx.strokeStyle = 'grey'
				NegativeBrushStrokes.forEach(stroke => {
				ctx.beginPath();
				stroke.forEach((point, index) => {
					ctx.lineWidth = point.brushSize/2.7;
					if (index === 0) {
					ctx.moveTo(point.x * canvasElement.width, point.y * canvasElement.height);
					}
					else {
					ctx.lineTo(point.x * canvasElement.width, point.y * canvasElement.height);
					}
				});
				ctx.stroke();
				});
			}

			

			

		
		
			const pointArray = Projection ? CoordLs2D : positiveKeypoints; // Select the point array based on the mode
			console.log("drawPoints called, pointArray: ");
			console.log(pointArray);

			// Re-draw the points
			pointArray.forEach((point, index) => {
			const x = point.x * canvasElement.width ;
			const y = point.y * canvasElement.height ;
		
			ctx.beginPath();
			ctx.arc(x, y, 5, 0, 2 * Math.PI);
			//   ctx.fillStyle = index === selectedPointIndex ? 'green' : 'red';
			ctx.fillStyle = pointColors[index % pointColors.length];
			ctx.fill();
			ctx.lineWidth = 1;
			//   ctx.strokeStyle = index === selectedPointIndex ? 'green' : 'red';
			ctx.strokeStyle = pointColors[index % pointColors.length];
			ctx.stroke();

				//Display confidences on the points
				ctx.font = "10px Arial";
				ctx.fillStyle = "black";
				
				if (confidences[index] !== undefined) {
					if (confidences[index] < 0.8) {
						ctx.fillStyle = "red";
						ctx.fillText(confidences[index].toFixed(2), x+10, y+10);
					}else{
						ctx.fillText(confidences[index].toFixed(2), x+10, y+10);
					}
					
				}
				
			});
			if (!Projection && negativeKeypoints.length > 0){
				negativeKeypoints.forEach((point, index) => {
					const x = point.x * canvasElement.width ;
					const y = point.y * canvasElement.height ;
				
					ctx.beginPath();
					ctx.arc(x, y, 5, 0, 2 * Math.PI);
					ctx.fillStyle = 'grey';
					ctx.fill();
					ctx.lineWidth = 1;
					ctx.strokeStyle = 'grey';
					ctx.stroke();

					//Display confidences on the points
					ctx.font = "10px Arial";
					ctx.fillStyle = "black";

					if (confidences[index + positiveKeypoints.length] !== undefined) {
						if (confidences[index + positiveKeypoints.length] < 0.8) {
							ctx.fillStyle = "red";
							ctx.fillText(confidences[index + positiveKeypoints.length].toFixed(2), x+10, y+10);
						}else{
							ctx.fillText(confidences[index + positiveKeypoints.length].toFixed(2), x+10, y+10);
						}
						
					}
					
				});
			}
	  };

	////////////////////// Brush /////////////////////////////////
	const [brushMode, setBrushMode] = useState(false);
	const [PositiveBrushStrokes, setPositiveBrushStrokes] = useState<{ x: number, y: number, brushSize: number }[][]>([]);
	const [NegativeBrushStrokes, setNegativeBrushStrokes] = useState<{ x: number, y: number, brushSize: number }[][]>([]);

	const [brushSize, setBrushSize] = useState(10);
	const handleBrushConfirm = () => {
		console.log("handleBrushConfirm called");
		console.log("brushStrokes: ", PositiveBrushStrokes);
	  };
	  
	  useEffect(() => {
		drawPoints();
	  }, [PositiveBrushStrokes, NegativeBrushStrokes]);
	///////////////////////////////////////////////////////////////////
	
	const handleMouseDown = (event: React.MouseEvent<HTMLCanvasElement>) => {
		console.log("handleMouseDown called, brushMode: ", brushMode);
		if (disableImageClick) return;
		const rect = canvasRef.current?.getBoundingClientRect();
		if (!rect) return;
		const canvasElement = canvasRef.current;
		if (!canvasElement) return;
		if (brushMode) {
		  const mouseX = event.clientX - rect.left;
		  const mouseY = event.clientY - rect.top;

		  const setBrushStrokes = mode === 'Add' ? setPositiveBrushStrokes : setNegativeBrushStrokes;
		
		  setBrushStrokes(prevStrokes => [...prevStrokes, [{ x: mouseX / canvasElement.width, y: mouseY / canvasElement.height, brushSize: brushSize }]]); // Start a new stroke
		  console.log('Mouse Down, start a new stroke at', { x: mouseX / canvasElement.width, y: mouseY/ canvasElement.height, brushSize: brushSize })
		  console.log('PositiveBrushStrokes:', PositiveBrushStrokes)
		  console.log('NegativeBrushStrokes:', NegativeBrushStrokes)
		//   drawPoints();
		} else {
			const mouseX = event.clientX - rect.left;
			const mouseY = event.clientY - rect.top;
			
			
			
			const pointArray = Projection ? CoordLs2D : (mode === 'Add' ? positiveKeypoints : negativeKeypoints); // Select the point array based on the mode
			const setPointArray = Projection ? setCoordLs2D : (mode === 'Add' ? setPositiveKeypoints : setNegativeKeypoints); // Set the point array based on the mode

			// Check if the click is on a point
			let pointFound = false;
			pointArray.forEach((point, index) => {
				const x = point.x * canvasElement.width ;
				const y = point.y * canvasElement.height ;
				const distance = Math.sqrt((x - mouseX) ** 2 + (y - mouseY) ** 2);
				if (distance < 10) { // 5 is the radius of the point
				setSelectedPointIndex(index);
				setIsDragging(true);
				pointFound = true;
				}
			});
			// If the point is not found, set the selected point index to null and add a new point
			if (!pointFound && !isDragging) {
				const coord = { x: mouseX / canvasElement.width, y: mouseY / canvasElement.height };
				setPointArray([...pointArray, coord]);
			}
			drawPoints();
		}

	};
	  
	const handleMouseMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
		if (disableImageClick) return;
		const setPointArray = Projection ? setCoordLs2D : (mode === 'Add' ? setPositiveKeypoints : setNegativeKeypoints); // Set the point array based on the mode
		if (selectedPointIndex === null) return;
		const rect = canvasRef.current?.getBoundingClientRect();
		if (!rect) return;
		
		

			const mouseX = event.clientX - rect.left;
			const mouseY = event.clientY - rect.top;
			
			const canvasElement = canvasRef.current;
			if (!canvasElement) return;
			
			const video_scale_factor = 256;
			
			const pointArray = Projection ? CoordLs2D : (mode === 'Add' ? positiveKeypoints : negativeKeypoints); // Select the point array based on the mode
			const updatedPoints = [...pointArray];
			updatedPoints[selectedPointIndex].x = mouseX  / canvasElement.width;
			updatedPoints[selectedPointIndex].y = mouseY  / canvasElement.height;
			
			setPointArray(updatedPoints);
			//console.log("updatedPoints: ", updatedPoints);

			drawPoints();
		
	};
	  
	// Listen to 3d coordinates change
	useEffect(() => {
		
		if (AUTO_PROCESS_COORDINATE){
			//console.log("Updated CoordLs2D.length:",CoordLs2D.length)
				if(CoordLs2D.length>=4 && !(nextFrameTime === 0) && CoordLs2D.length === coordinates3D.length){
					handleProcessCoordinates();
				}
				//console.log("Updated CoordLs2D:", CoordLs2D);
				//console.log("Updated coordinates3D:", coordinates3D);
		}
		
	}, [coordinates3D])

	const handleMouseUp = (event: React.MouseEvent<HTMLCanvasElement>) => {
		if (disableImageClick) return;
		const canvasElement = canvasRef.current;
		if (!canvasElement) return;

		console.log("handleMouseUp called");
		if (brushMode && (PositiveBrushStrokes.length > 0 || NegativeBrushStrokes.length > 0)) {
			const rect = canvasRef.current?.getBoundingClientRect();
			if (!rect) return;
			const BrushStrokes = mode === 'Add' ? PositiveBrushStrokes : NegativeBrushStrokes;
			const setBrushStrokes = mode === 'Add' ? setPositiveBrushStrokes : setNegativeBrushStrokes;
			const mouseX = event.clientX - rect.left;
			const mouseY = event.clientY - rect.top;
			const currentStroke = BrushStrokes[BrushStrokes.length - 1];
			console.log('Mouse Up, add', { x: mouseX, y: mouseY, brushSize: brushSize })
			console.log('currentStroke:', currentStroke)
			console.log('PositiveCrushStrokes:', PositiveBrushStrokes)
			console.log('NegativeCrushStrokes:', NegativeBrushStrokes)
			setBrushStrokes(prevStrokes => [...prevStrokes.slice(0, -1), [...currentStroke, { x: mouseX/ canvasElement.width, y: mouseY/ canvasElement.height, brushSize: brushSize }]]);
			drawPoints();
		} else {

			if (AUTO_PROCESS_COORDINATE){
				if(Projection){
					//console.log("Updated CoordLs2D.length:",CoordLs2D.length)
					if(CoordLs2D.length>=4 && !(nextFrameTime === 0) && CoordLs2D.length === coordinates3D.length){
						handleProcessCoordinates();
					}
					//console.log("Updated CoordLs2D:", CoordLs2D);
					//console.log("Updated coordinates3D:", coordinates3D);
				}else{
					//console.log("Updated positiveKeypoints.length:",positiveKeypoints.length)
					if (positiveKeypoints.length > 0 || negativeKeypoints.length > 0){
						handleProcessCoordinates();
					}
					//console.log("Updated positiveKeypoints:", positiveKeypoints);
					//console.log("Updated negativeKeypoints:", negativeKeypoints);
				}
			}

			//console.log('currentFrameTime:')
			//console.log(currentFrameTime)
			if (selectedPointIndex === null) return;
			setSelectedPointIndex(null);
			setIsDragging(false); 
			drawPoints();
		}
	};

	/////////////////////////////End of async functions for moving points on canvas///////////////////////////
	// useEffect(() => {
	// 	//console.log('currentFrameTime has been updated:', currentFrameTime);
	//   }, [currentFrameTime]);
	//   useEffect(() => {
	// 	//console.log('positiveKeypoints has been updated:', positiveKeypoints);
	//   }, [positiveKeypoints]);
	//   useEffect(() => {
	// 	//console.log('CoordLs2D has been updated:', CoordLs2D);
	//   }
	//   , [CoordLs2D]);
	//   useEffect(() => {
	// 	//console.log('negativeKeypoints has been updated:', negativeKeypoints);
	//   }
	//   , [negativeKeypoints]);
	//   useEffect(() => {
	// 	//console.log('confidences has been updated:', confidences);
	//   }
	//   , [confidences]);

	///////////// async function for run TAPIR to do frame propogation//////////////////////

	const trackingPointsInit = async () => {

		// Append the waiting GIF to the image-container-2 div
		const imageContainer = document.getElementById('image-container-2');
		const canvas = document.getElementById('canvas');
		const loader = document.getElementById('loader-middle');

		if (imageContainer) {
			if (canvas) {
			canvas.style.display = 'none';
			}
			if (loader) {
				loader.style.display = 'block';
			  }
			setDisableImageClick(true);

		}
		const combinePoints = positiveKeypoints.concat(negativeKeypoints);
		//console.log("combinePoints: ", combinePoints)

		const pointsCoordinates = Projection ? CoordLs2D : combinePoints;
		//console.log("handleProcessCoordinates called");
		const videoElement = videoRef.current;
		if (!videoElement) return;
		//console.log("videoElement.currentTime: ", videoElement.currentTime);
		if(!canvasImage) return;
	  
		if (POINTS_TRACKING) {
			fetch('http://'+TAPIR_host + ':' + TAPIR_port + '/init-points' , {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},

				body: JSON.stringify({  "user": user, 'image': canvasImage.src,'videoPath': selectedVideo, '2d-coordinates': pointsCoordinates, "currentFrameTime": videoElement.currentTime })
			})
			.then(response => response.json())
			.then(data => {
				setPointsInitilized(true);
				//console.log(data)
				// Remove the waiting GIF after receiving the response
				if (canvas) {
				canvas.style.display = 'block';
				}
				if (loader) {
				loader.style.display = 'none';
				}
				setDisableImageClick(false)
				setMessages(data.messages);
				// alert(data.messages);

			})
			.catch(error => {
				console.error('Error making POST request:', error);
				if (error.message.includes('ERR_CONNECTION_REFUSED') || error.message.includes('Failed to fetch')) {
				  console.log('Connection refused. Refreshing page...');
				  window.location.reload();
				}
				console.error('Error:', error);
			
				// Hide the waiting GIF after receiving the response
				if (loader) {
				loader.style.display = 'none';
				}
			});
		}

	  };

		
	const transformToCoordinates = (array: number[][]): Coordinate[] => {
		const video_scale_factor = 256;
		return array.map((item) => {
		  return { x: item[0]/video_scale_factor, y: item[1]/video_scale_factor };
		});
	  };

	////
	const handlePartNotAbleToAnnotate = () => {
		updateProgressBar()
		handleClean()
		const videoElement = videoRef.current;
		if (!videoElement) return;
		videoElement.pause();

		const fps = 1;
		const frameDuration = 1 / fps;

		// Set the video element's current time to the next frame time
		videoElement.currentTime = currentFrameTime + frameDuration; // Move to the next frame
		setNextFrameTime(prevTime => prevTime + frameDuration); //Move to the next frame


		// Add a one-time event listener for 'seeked'
		videoElement.addEventListener('seeked', async () => {
		const canvasElement = canvasRef.current;


		if (!videoElement || !canvasElement) {
			console.error("Video or canvas element is not available");
			return;
		}

		const ctx = canvasElement.getContext("2d");
		if (!ctx) {
			console.error("Unable to get 2D context from canvas");
			return;
		}

		setNextFrameTime(prevTime => prevTime + 1); //Move to the next frame

		// Get the image from backend and draw it on the canvas
		//console.log('start trying to come back to the previous anno')
        try {
			setHasNewAnnotation(false)
            const response = await fetch(`http://${host}:${port}/part-not-able-to-present`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    "user": user,
                    "videoPath": selectedVideo,
                    "Category": selectedCategory,
                    "SubCategory": selectedSubCategory,
                    "Step": selectedStep,
                    'currentModelFilePaths': currentModelFilePaths,
                    'nextFrameTime': nextFrameTime,
					'Projection': Projection,
					'NotAbleToAnnotate': true,
					'Message': null,
					'ReportMessage': false,
                })
            });

            const data = await response.json();
			if (data.messages){
				setMessages(data.messages);
			}
			if (data.error){
				//console.log("Error received:", data);
				alert(data.error);
				if (data.error === "You have reached the end of the video."){
					setNextFrameTime(0);
					setCurrentFrameTime(0);
					setAnnotationForCurrentFrameExist(false)
					
					return;
				}
			}
			if(data.imagePath != null){
				setAnnotationForCurrentFrameExist(true)
				const unique_suffix = "?t=" + new Date().getTime();
				setCurrentUpperImage(data.imagePath + unique_suffix);
			}else{
				setAnnotationForCurrentFrameExist(false)
				setCurrentUpperImage('');
				
			}
		
			//Set current frame time and set video element's current time
			setCurrentFrameTime(data.currentFrameTime);
			videoElement.currentTime = data.currentFrameTime;

			if (data.base64Image != null){
				alert('Moved to the next frame! Success!')
				updateProgressBar()
			}
			else{
				alert('Moved to the next failed!')
			}
			//console.log("Data received:", data);
			const base64Image = data.base64Image;

			// Reshape the image to fit the canvas
			const image = new Image();
			image.src = 'data:image/png;base64,' +base64Image;
			image.onerror = async function(error) {
				console.error("Error loading the image:", error);
			};
			
			image.onload = () => {
				// Set canvas dimensions to match the video
				const container = document.getElementById("middle-part");
				if (container) {
					const containerWidth = container.clientWidth;
					const containerHeight = container.clientHeight;

					// Make sure the image still in the original aspect ratio
					const scale = Math.min(containerWidth / image.width, containerHeight / image.height);
					const scaledWidth = image.width * scale;
					const scaledHeight = image.height * scale;
					const left = (containerWidth - scaledWidth) / 2;
					const top = (containerHeight - scaledHeight) / 2;
					//console.log(`left, top: (${left}, ${top})`);
					//console.log(`scaledWidth, scaledHeight: (${scaledWidth}, ${scaledHeight})`);

					canvasElement.width = scaledWidth;
					canvasElement.height = scaledHeight;
					//console.log(`containerWidth, containerHeight: (${containerWidth}, ${containerHeight})`);
				}

				// Clean the canvas first
				if (ctx) {
					ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
				}
				// Draw the current video frame on the canvas
				ctx.drawImage(image, 0, 0, canvasElement.width, canvasElement.height);
				//console.log("image.src: ", image.src);
				setCanvasImage(image);
				//console.log("image: ", image);
			};
		
		
        } catch (error) {
			if (error instanceof Error) {
				if (error.message.includes('ERR_CONNECTION_REFUSED') || error.message.includes('Network request failed') || error.message.includes('Failed to fetch')) {
					// Here you can handle the specific error
					// For example, by refreshing the page:
					console.log('Connection refused. Refreshing page...');
					window.location.reload();
				}
			}
            console.error('Error:', error);
        }


		const canvasImage = new Image();
		canvasImage.src = canvasElement.toDataURL("image/png");
		const canvasURL = canvasElement.toDataURL("image/png")
		//console.log("canvasURL: ", canvasURL);
		setCanvasImage(canvasImage);
		//console.log("canvasImage: ", canvasImage);
		//console.log("videoElement: ", videoElement);
	
		
		canvasElement.style.display = 'block';
	}  , { once: true });
		
	}
	const handlePartNotShowUp = () => {
		updateProgressBar()
		handleClean()
		const videoElement = videoRef.current;
		if (!videoElement) return;
		videoElement.pause();

		const fps = 1;
		const frameDuration = 1 / fps;

		// Set the video element's current time to the next frame time
		videoElement.currentTime = currentFrameTime + frameDuration; // Move to the next frame
		setNextFrameTime(prevTime => prevTime + frameDuration); //Move to the next frame


		// Add a one-time event listener for 'seeked'
		videoElement.addEventListener('seeked', async () => {
		const canvasElement = canvasRef.current;


		if (!videoElement || !canvasElement) {
			console.error("Video or canvas element is not available");
			return;
		}

		const ctx = canvasElement.getContext("2d");
		if (!ctx) {
			console.error("Unable to get 2D context from canvas");
			return;
		}

		setNextFrameTime(prevTime => prevTime + 1); //Move to the next frame

		// Get the image from backend and draw it on the canvas
		//console.log('start trying to come back to the previous anno')
        try {
			setHasNewAnnotation(false)
            const response = await fetch(`http://${host}:${port}/part-not-able-to-present`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    "user": user,
                    "videoPath": selectedVideo,
                    "Category": selectedCategory,
                    "SubCategory": selectedSubCategory,
                    "Step": selectedStep,
                    'currentModelFilePaths': currentModelFilePaths,
                    'nextFrameTime': nextFrameTime,
					'Projection': Projection,
					'NotAbleToAnnotate': false,
					'Message': null,
					'ReportMessage': false,
                })
            });

            const data = await response.json();
			if (data.messages){
				setMessages(data.messages);
			}
			if (data.error){
				//console.log("Error received:", data);
				alert(data.error);
				if (data.error === "You have reached the end of the video."){
					setNextFrameTime(0);
					setCurrentFrameTime(0);
					setAnnotationForCurrentFrameExist(false)
					
					return;
				}
			}
			if(data.imagePath != null){
				setAnnotationForCurrentFrameExist(true)
				const unique_suffix = "?t=" + new Date().getTime();
				setCurrentUpperImage(data.imagePath + unique_suffix);
			}else{
				setAnnotationForCurrentFrameExist(false)
				setCurrentUpperImage('');
			}
		
			//Set current frame time and set video element's current time
			setCurrentFrameTime(data.currentFrameTime);
			videoElement.currentTime = data.currentFrameTime;

			//console.log("Data received:", data);
			const base64Image = data.base64Image;

			if (data.base64Image != null){
				alert('Moved to the next frame! Success!')
				updateProgressBar()
			}
			else{
				alert('Moved to the next failed!')
			}

			// Reshape the image to fit the canvas
			const image = new Image();
			image.src = 'data:image/png;base64,' +base64Image;
			image.onerror = async function(error) {
				console.error("Error loading the image:", error);
			};
			
			image.onload = () => {
				// Set canvas dimensions to match the video
				const container = document.getElementById("middle-part");
				if (container) {
					const containerWidth = container.clientWidth;
					const containerHeight = container.clientHeight;

					// Make sure the image still in the original aspect ratio
					const scale = Math.min(containerWidth / image.width, containerHeight / image.height);
					const scaledWidth = image.width * scale;
					const scaledHeight = image.height * scale;
					const left = (containerWidth - scaledWidth) / 2;
					const top = (containerHeight - scaledHeight) / 2;
					//console.log(`left, top: (${left}, ${top})`);
					//console.log(`scaledWidth, scaledHeight: (${scaledWidth}, ${scaledHeight})`);

					canvasElement.width = scaledWidth;
					canvasElement.height = scaledHeight;
					//console.log(`containerWidth, containerHeight: (${containerWidth}, ${containerHeight})`);
				}

				// Clean the canvas first
				if (ctx) {
					ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
				}
				// Draw the current video frame on the canvas
				ctx.drawImage(image, 0, 0, canvasElement.width, canvasElement.height);
				//console.log("image.src: ", image.src);
				setCanvasImage(image);
				//console.log("image: ", image);
			};
		
		
        } catch (error) {
			if (error instanceof Error) {
				if (error.message.includes('ERR_CONNECTION_REFUSED') || error.message.includes('Network request failed') || error.message.includes('Failed to fetch')) {
					// Here you can handle the specific error
					// For example, by refreshing the page:
					console.log('Connection refused. Refreshing page...');
					window.location.reload();
				}
			}
            console.error('Error:', error);
        }


		const canvasImage = new Image();
		canvasImage.src = canvasElement.toDataURL("image/png");
		const canvasURL = canvasElement.toDataURL("image/png")
		//console.log("canvasURL: ", canvasURL);
		setCanvasImage(canvasImage);
		//console.log("canvasImage: ", canvasImage);
		//console.log("videoElement: ", videoElement);
	
		
		canvasElement.style.display = 'block';
	}  , { once: true });
		
	}

	const handleBack = () => {
		updateProgressBar()
		handleClean()
		const videoElement = videoRef.current;
		// Pause the video element
		if (!videoElement) return;
		//console.log("videoElement.currentTime: ", videoElement.currentTime);
		videoElement.pause();

	
		const fps = 1;
		const frameDuration = 1 / fps;

		// Log the current frame time
		//console.log('currentFrameTime:')
		//console.log(currentFrameTime)
		
		// Set the video element's current time to the next frame time
		videoElement.currentTime = currentFrameTime + frameDuration; // Move to the next frame
		setNextFrameTime(prevTime => prevTime + frameDuration); //Move to the next frame
		//console.log("videoElement.currentTime")
		//console.log(videoElement.currentTime)


		// Add a one-time event listener for 'seeked'
		videoElement.addEventListener('seeked', async () => {
		const canvasElement = canvasRef.current;


		if (!videoElement || !canvasElement) {
			console.error("Video or canvas element is not available");
			return;
		}

		const ctx = canvasElement.getContext("2d");
		if (!ctx) {
			console.error("Unable to get 2D context from canvas");
			return;
		}

		// Get the image from backend and draw it on the canvas
		//console.log('start trying to come back to the previous anno')
        try {
			setHasNewAnnotation(false)
            const response = await fetch(`http://${host}:${port}/back-to-prev-anno-frame`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    "user": user,
                    "videoPath": selectedVideo,
                    "Category": selectedCategory,
                    "SubCategory": selectedSubCategory,
                    "Step": selectedStep,
                    'currentModelFilePaths': currentModelFilePaths,
                    'nextFrameTime': nextFrameTime,
					'Projection': Projection,
                })
            });

            const data = await response.json();
			if (data.messages){
				setMessages(data.messages);
			}
			if (data.error){
				//console.log("Error received:", data);
				alert(data.error);
				if (data.error === "You have reached the end of the video."){
					setNextFrameTime(0);
					setCurrentFrameTime(0);
					setAnnotationForCurrentFrameExist(false)
					
					return;
				}
			}
			if(data.imagePath != null){
				setAnnotationForCurrentFrameExist(true)
				const unique_suffix = "?t=" + new Date().getTime();
				setCurrentUpperImage(data.imagePath + unique_suffix);
			}else{
				setAnnotationForCurrentFrameExist(false)
				setCurrentUpperImage('');	
				alert('No previous annotated frame!')
			}
		
			//Set current frame time and set video element's current time
			setCurrentFrameTime(data.currentFrameTime);
			setNextFrameTime(data.currentFrameTime);
			videoElement.currentTime = data.currentFrameTime;

			//console.log("Data received:", data);
			const base64Image = data.base64Image;

			if (data.base64Image != null){
				alert('Moved to the frame! Success!')
				updateProgressBar()
			}else{
				alert('Moved to the frame failed!')
			}

			// Reshape the image to fit the canvas
			const image = new Image();
			image.src = 'data:image/png;base64,' +base64Image;
			image.onerror = async function(error) {
				console.error("Error loading the image:", error);
			};
			
			image.onload = () => {
				// Set canvas dimensions to match the video
				const container = document.getElementById("middle-part");
				if (container) {
					const containerWidth = container.clientWidth;
					const containerHeight = container.clientHeight;

					// Make sure the image still in the original aspect ratio
					const scale = Math.min(containerWidth / image.width, containerHeight / image.height);
					const scaledWidth = image.width * scale;
					const scaledHeight = image.height * scale;
					const left = (containerWidth - scaledWidth) / 2;
					const top = (containerHeight - scaledHeight) / 2;
					//console.log(`left, top: (${left}, ${top})`);
					//console.log(`scaledWidth, scaledHeight: (${scaledWidth}, ${scaledHeight})`);

					canvasElement.width = scaledWidth;
					canvasElement.height = scaledHeight;
					//console.log(`containerWidth, containerHeight: (${containerWidth}, ${containerHeight})`);
				}

				// Clean the canvas first
				if (ctx) {
					ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
				}
				// Draw the current video frame on the canvas
				ctx.drawImage(image, 0, 0, canvasElement.width, canvasElement.height);
				//console.log("image.src: ", image.src);
				setCanvasImage(image);
				//console.log("image: ", image);
			};
		
		
        } catch (error) {
			if (error instanceof Error) {
				if (error.message.includes('ERR_CONNECTION_REFUSED') || error.message.includes('Network request failed') || error.message.includes('Failed to fetch')) {
					// Here you can handle the specific error
					// For example, by refreshing the page:
					console.log('Connection refused. Refreshing page...');
					window.location.reload();
				}
			}
            console.error('Error:', error);
        }


		const canvasImage = new Image();
		canvasImage.src = canvasElement.toDataURL("image/png");
		const canvasURL = canvasElement.toDataURL("image/png")
		//console.log("canvasURL: ", canvasURL);
		setCanvasImage(canvasImage);
		//console.log("canvasImage: ", canvasImage);
		//console.log("videoElement: ", videoElement);
	
		
		canvasElement.style.display = 'block';
	}  , { once: true });
		
	}
	


	const updateProgressBar = () => {
		fetch('http://' + host + ':' + port + '/get-progress-bar', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json'
			},
			body: JSON.stringify({ 'category': selectedCategory, 'name': selectedSubCategory, 'step_id': selectedStep, 'videoPath': selectedVideo, 'user': user, 'Projection': Projection, 'currentModelFilePaths': currentModelFilePaths})
		})
		.then(response => response.json())
			.then(data => {
				if (data.error){
					//console.log("Error received:", data);
					alert(data.error);
				}
				setTotalCurrPartCurrStep(data.total_annotation_needed_for_curr_part_curr_step);
				setAnnotatedCurrPartCurrStep(data.num_of_annotated_for_curr_part_curr_step);

				setTotalCurrStepCurrVideo(data.total_annotation_needed_for_curr_step_curr_video);
				setAnnotatedCurrStepCurrVideo(data.num_of_annotated_for_curr_step_curr_video);

				setTotalCurrStep(data.total_annotation_needed_for_curr_step);
				setAnnotatedCurrStep(data.num_of_annotated_for_curr_step);

				setTotalCurrVideo(data.total_annotation_needed_for_curr_video);
				setAnnotatedCurrVideo(data.num_of_annotated_for_curr_video);

				setTotalCurrFurniture(data.total_annotation_needed_for_curr_furniture);
				setAnnotatedCurrFurniture(data.num_of_annotated_for_curr_furniture)

				setTotalAllFurniture(data.total_annotation_needed_for_all_furniture)
				setAnnotatedAllFurniture(data.num_of_annotated_for_all_furniture)




			}
			)
			.catch(error => {
				console.error('Error making POST request:', error);
				if (error.message.includes('ERR_CONNECTION_REFUSED') || error.message.includes('Failed to fetch')) {
				  console.log('Connection refused. Refreshing page...');
				  window.location.reload();
				}
			});
		}

	async function handleNextFrame() {
		updateProgressBar()
		//console.log('call next frame')
		
		// if points are initialized, run TAPIR
		if (! (nextFrameTime === 0) && !pointsInitilized && POINTS_TRACKING){
			alert("Please initialize the points first.");
			return;
		}
		//console.log('nextFrameTIme',nextFrameTime )
		if (! (nextFrameTime === 0) ){

			if(hasNewAnnotation){
				//Raise an alert to remind the user to double check current annotation, if user click yes then continue
				if (annotationForCurrentFrameExist){ //annotationForCurrentFrameExist is false if /next-frame return null for image path or there is no new annotation (since it's false by default)
					const r = window.confirm("Previous annotation exists for this frame. Do you want to overwrite it? Click 'Cancel' to move to the next frame without overwriting");
					if(r){
						if (Projection){
							await handleSaveProjection(false);
						}else{
							await handleSaveMask();
						}
						//console.log('Yes clicked')

					}else{
						//console.log('No clicked. Annotation not saved. Move to the next frame!');
					}
				}else{
					if (Projection){
					await handleSaveProjection(false);
					}else{
					await handleSaveMask();
					}
				}
			}else{ //(!hasNewAnnotation){

				const r2 =window.confirm("There is no annotation for this frame. Do you want to skip current frame?");
				if(r2){
					//console.log('No new annotation. Skip Current Frame. Move to the next frame!');
				}else{
					//console.log('No new annotation. Stay at current frame.');
					return;

				}
			}

		}
		if (selectedFrame === ''){
			alert("Please select a frame first.");
			return;
		}
		if (currentModelFilePaths.length === 0){
			alert("Please select a part first.");
			return;
		}
		
		if (!POINTS_TRACKING){
			handleCleanKeepPoints()

		}
		// else{
		// 	handleClean()
			
		// }
		const videoElement = videoRef.current;
		// Pause the video element
		if (!videoElement) return;
		//console.log("videoElement.currentTime: ", videoElement.currentTime);
		videoElement.pause();

	
		const fps = 1;
		const frameDuration = 1 / fps;

		// Log the current frame time
		//console.log('currentFrameTime:')
		//console.log(currentFrameTime)


		// Set the video element's current time to the next frame time
		videoElement.currentTime = currentFrameTime + frameDuration; // Move to the next frame
		setNextFrameTime(prevTime => prevTime + frameDuration); //Move to the next frame
		//console.log("videoElement.currentTime")
		//console.log(videoElement.currentTime)


		// Add a one-time event listener for 'seeked'
		// videoElement.addEventListener('seeked', async () => {
		const canvasElement = canvasRef.current;


		if (!videoElement || !canvasElement) {
			console.error("Video or canvas element is not available");
			return;
		}

		const ctx = canvasElement.getContext("2d");
		if (!ctx) {
			console.error("Unable to get 2D context from canvas");
			return;
		}

		// Get the image from backend and draw it on the canvas
		//console.log('start trying to get next frame')
        try {
			setHasNewAnnotation(false)
            const response = await fetch(`http://${host}:${port}/next-frame`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    "user": user,
                    "videoPath": selectedVideo,
                    "Category": selectedCategory,
                    "SubCategory": selectedSubCategory,
                    "Step": selectedStep,
                    'currentModelFilePaths': currentModelFilePaths,
                    'nextFrameTime': nextFrameTime,
					'Projection': Projection,
                })
            });

            const data = await response.json();
			if (data.messages){
				setMessages(data.messages);
			}
			if (data.error){
				//console.log("Error received:", data);
				alert(data.error);
				if (data.error === "You have reached the end of the video."){
					setNextFrameTime(0);
					setCurrentFrameTime(0);
					setAnnotationForCurrentFrameExist(false)
					
					return;
				}
			}
			if(data.imagePath != null){
				setAnnotationForCurrentFrameExist(true)
				const unique_suffix = "?t=" + new Date().getTime();
				setCurrentUpperImage(data.imagePath + unique_suffix);
			}else{
				setAnnotationForCurrentFrameExist(false)
				setCurrentUpperImage('');
			}
		
			//Set current frame time and set video element's current time
			setCurrentFrameTime(data.currentFrameTime);
			videoElement.currentTime = data.currentFrameTime;

			//console.log("Data received:", data);
			const base64Image = data.base64Image;

			// Reshape the image to fit the canvas
			const image = new Image();
			image.src = 'data:image/png;base64,' +base64Image;
			image.onerror = async function(error) {
				console.error("Error loading the image:", error);
			};
			
			image.onload = () => {
				// Set canvas dimensions to match the video
				const container = document.getElementById("middle-part");
				if (container) {
					const containerWidth = container.clientWidth;
					const containerHeight = container.clientHeight;

					// Make sure the image still in the original aspect ratio
					const scale = Math.min(containerWidth / image.width, containerHeight / image.height);
					const scaledWidth = image.width * scale;
					const scaledHeight = image.height * scale;
					const left = (containerWidth - scaledWidth) / 2;
					const top = (containerHeight - scaledHeight) / 2;
					//console.log(`left, top: (${left}, ${top})`);
					//console.log(`scaledWidth, scaledHeight: (${scaledWidth}, ${scaledHeight})`);

					canvasElement.width = scaledWidth;
					canvasElement.height = scaledHeight;
					//console.log(`containerWidth, containerHeight: (${containerWidth}, ${containerHeight})`);
				}

				// Clean the canvas first
				if (ctx) {
					ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
				}
				// Draw the current video frame on the canvas
				ctx.drawImage(image, 0, 0, canvasElement.width, canvasElement.height);
				//console.log("image.src: ", image.src);
				setCanvasImage(image);
				//console.log("image: ", image);
			};
		
		
        } catch (error) {

            console.error('Error:', error);
			if (error instanceof Error) {
				if (error.message.includes('ERR_CONNECTION_REFUSED') || error.message.includes('Network request failed') || error.message.includes('Failed to fetch')) {
					// Here you can handle the specific error
					// For example, by refreshing the page:
					console.log('Connection refused. Refreshing page...');
					window.location.reload();
				}
			}
        }


		const canvasImage = new Image();
		canvasImage.src = canvasElement.toDataURL("image/png");
		const canvasURL = canvasElement.toDataURL("image/png")
		//console.log("canvasURL: ", canvasURL);
		setCanvasImage(canvasImage);
		//console.log("canvasImage: ", canvasImage);
		//console.log("videoElement: ", videoElement);
	
		
		// If on the first frame 
		if (nextFrameTime === 0 && canvasElement) {
			// Show the canvas
			canvasElement.style.display = 'block';

		}else {
			if(POINTS_TRACKING){
				// Get prediction, segmentation/pose-estimation

				const imageContainer = document.getElementById('image-container-2');
				const canvas = document.getElementById('canvas');
				const loader = document.getElementById('loader-middle');

				if (imageContainer) {
					if (canvas) {
						canvas.style.display = 'none';
					}
					if (loader) {
						loader.style.display = 'block';
					}
				}
				setDisableImageClick(true);


				const positiveLength = positiveKeypoints.length;


				try {
					const response = await fetch('http://'+host + ':' + port + '/get-predictions' , {
						method: 'POST',
						headers: {
						'Content-Type': 'application/json'
						},

						body: JSON.stringify({ 
						"user": user, 
						'currentFrameTime': videoElement.currentTime,
						'Category': selectedCategory, 
						'SubCategory': selectedSubCategory, 
						'Object': selectedStep, 
						'video': selectedVideo, 
						'image': canvasURL, 
						'Projection': Projection,
						"3d-coordinates": coordinates3D
						})
					})
					const data = await response.json();

					//console.log('user :', user)
					// Remove the waiting GIF after receiving the response
					if (canvas) {canvas.style.display = 'block';}
					if (loader) {loader.style.display = 'none';}		
					setDisableImageClick(false)

					//console.log("Prediction: ", data)
					// Handle response data here
					if (data.error){
						//console.log("Error received:", data);
						alert(data.error);
					}
					if (data.messages){
						setMessages(data.messages);
					}
					const pointsCoordinates = data.pointsCoordinates;
					const transformedPoints = transformToCoordinates(pointsCoordinates);
					// Normalize the coordinates
					const confidences = data.confidences;
					setConfidences(confidences);
					//console.log(confidences)


					if(!Projection){
						//console.log("positiveLength: ", positiveLength);
						// Set positive keypoints to the first positiveLength points
						setPositiveKeypoints(transformedPoints.slice(0, positiveLength));
						//console.log('transformedPoints.slice(0, positiveLength)', transformedPoints.slice(0, positiveLength))
						// Set negative keypoints to the rest points
						setNegativeKeypoints(transformedPoints.slice(positiveLength));
						//console.log('transformedPoints.slice(positiveLength)', transformedPoints.slice(positiveLength))

					}else{
						setCoordLs2D(transformedPoints);
					}
					//console.log("transformedPoints: ");
					//console.log(transformedPoints);

					//console.log('canvas width, height', canvasElement.width, canvasElement.height);
					//console.log('video width, height', videoElement.videoWidth, videoElement.videoHeight);
					// handleProcessCoordinates();
					drawPoints();
					////////////////////////////////////////
					///////////////////



					// Load the waiting GIF
					const imageContainer = document.getElementById('image-container-2');
					const upperImage = document.getElementById('upper-image');
					const loader_upper = document.getElementById('loader-upper');
					const upperCanvasImg = document.getElementById('upper-canvas-img');





					// setMessages([]);
					if (AUTO_PROCESS_COORDINATE){
						if (imageContainer) {
							if (upperImage) {
								upperImage.style.display = 'none';
							}
							if (upperCanvasImg) {
								upperCanvasImg.style.display = 'none';
							}

							if (loader_upper) {
								loader_upper.style.display = 'block';
							}
						}
						setDisableImageClick(true);


						//console.log("handleProcessCoordinates called");
						//console.log("videoElement.currentTime: ", videoElement.currentTime);


						if (Projection){ // Pass both 2D and 3D coordinates	

						try{
						// Send coordinates to backend
						const response = await fetch('http://'+ host+':'+port+'/pose-estimation', {
						method: 'POST',
						headers: {
						'Content-Type': 'application/json'
						},
						body: JSON.stringify({  "user": user, "2d-coordinates": transformedPoints, "3d-coordinates": coordinates3D
						, "Category": selectedCategory, "SubCategory": selectedSubCategory, "Object": selectedStep,
						'currentModelFilePaths': currentModelFilePaths})

						})

						const data = await response.json();


						if (upperImage) {
						upperImage.style.display = 'block';
						}
						if (loader_upper) {
						loader_upper.style.display = 'none';
						}
						if (upperCanvasImg) {
						upperCanvasImg.style.display = 'block';
						}
						setDisableImageClick(false);


						//console.log("Data received:", data);

						if (data.messages){
						setMessages(data.messages);
						}

						const unique_suffix = "?t=" + new Date().getTime();
						setCurrentUpperImage(data.imagePath[0] + unique_suffix);
						setImages(data.imagePath);	
						setHasNewAnnotation(true)		
						console.log('data.imagePath', data.imagePath)		
						

						}catch(error){
							if (error instanceof Error) {
								if (error.message.includes('ERR_CONNECTION_REFUSED') || error.message.includes('Network request failed') || error.message.includes('Failed to fetch')) {
									// Here you can handle the specific error
									// For example, by refreshing the page:
									console.log('Connection refused. Refreshing page...');
									window.location.reload();
								}
							}
						//console.log("Error received:", data);
						// alert(data.error);
						if (data.error === "No part selected. Please select at least one part for segmentation."){
						alert("No part selected. Please select at least one part for segmentation.");
						}
						// 'SolvePnP failed.'
						else if(data.error === 'SolvePnP failed.'){
						alert('SolvePnP failed. Please check if the selected points are correct.');
						}else if(data.error ==='Not enough keypoints. Please select at least 4 keypoints.'){
						alert('Not enough keypoints. Please select at least 4 keypoints.');
						}else if(data.error ==='Keypoints shape mismatch.'){
						alert('Keypoints shape mismatch. Please check if the selected points are correct.');
						}else{
						setMessages(data.error);
						}

						setCurrentUpperImage(''); // Clear the upper image
						}


						}else{ // Pass only positive and negative keypoints, and display the segmentation result at the original image

						try {
						const response = await fetch('http://'+ host+':'+port+'/segmentation', {
						method: 'POST',
						headers: {
						'Content-Type': 'application/json'
						},
						body: JSON.stringify({  "user": user, "positive-keypoints": transformedPoints.slice(0, positiveLength), "negative-keypoints": transformedPoints.slice(positiveLength),
						"Category": selectedCategory, "SubCategory": selectedSubCategory, "Object": selectedStep, 'currentModelFilePaths': currentModelFilePaths, 'PositiveBrushStrokes': PositiveBrushStrokes, 'NegativeBrushStrokes': NegativeBrushStrokes,
					'with_prev_mask':false})

						})
						const data = await response.json();
						//console.log("Data received:", data);
						//console.log('canvasURL',canvasURL)
						// handleSaveMask();

						if (upperImage) {upperImage.style.display = 'block';}
						if (loader_upper) {loader_upper.style.display = 'none';}
						if (upperCanvasImg) {upperCanvasImg.style.display = 'block';}
						setDisableImageClick(false);


						if (data.messages){setMessages(data.messages);}
						//console.log("Data received:", data);
						//console.log(data);
						const unique_suffix = "?t=" + new Date().getTime();
						setCurrentUpperImage(data.imagePath + unique_suffix);
						//console.log("current upper image: ");
						//console.log(data.imagePath + unique_suffix);

						setHasNewAnnotation(true)

						} catch (error) {
							if (error instanceof Error) {
								if (error.message.includes('ERR_CONNECTION_REFUSED') || error.message.includes('Network request failed') || error.message.includes('Failed to fetch')) {
									// Here you can handle the specific error
									// For example, by refreshing the page:
									console.log('Connection refused. Refreshing page...');
									window.location.reload();
								}
						}
						//console.log("Error received:", data);
						if (data.error === "No part selected. Please select at least one part for segmentation."){
						alert("No part selected. Please select at least one part for segmentation.");
						}
						}
						}

					}else{
						if(Projection){
							// Send coordinates to backend
							const response = await fetch('http://'+ host+':'+port+'/create-pose-estimation-data', {
							method: 'POST',
							headers: {
							'Content-Type': 'application/json'
							},
							body: JSON.stringify({  "user": user, "2d-coordinates": transformedPoints, "3d-coordinates": coordinates3D
							, "Category": selectedCategory, "SubCategory": selectedSubCategory, "Object": selectedStep,
							'currentModelFilePaths': currentModelFilePaths})

							})	
						}else{
							const response = await fetch('http://'+ host+':'+port+'/create-segmentation_data', {
							method: 'POST',
							headers: {
							'Content-Type': 'application/json'
							},
							body: JSON.stringify({  "user": user, "positive-keypoints": transformedPoints.slice(0, positiveLength), "negative-keypoints": transformedPoints.slice(positiveLength),
							"Category": selectedCategory, "SubCategory": selectedSubCategory, "Object": selectedStep, 'currentModelFilePaths': currentModelFilePaths})

							})


						}
						setDisableImageClick(false);
					}

				} catch (error) {
					if (error instanceof Error) {
						if (error.message.includes('ERR_CONNECTION_REFUSED') || error.message.includes('Network request failed') || error.message.includes('Failed to fetch')) {
							// Here you can handle the specific error
							// For example, by refreshing the page:
							console.log('Connection refused. Refreshing page...');
							window.location.reload();
						}
					}
				console.error('Error:', error);
				}
			}else{
				drawPoints();
			}
		}
	
	// }  , { once: true });
		
	}

	

	
	
	
	////
	
	  const handleProjectionToggleButtonClick = () => {
		// Reset annotatinon data
		// Reset the video
		const event = {
			target: {
			value: '', 
			},
		} as unknown as React.ChangeEvent<HTMLSelectElement>;

		setSelectedVideo('');
		handleVideoChange(event);

		
		// hide the canvas
		const canvas = document.getElementById('canvas');
		if (canvas) {
			canvas.style.display = 'none';
		}
		
		

		//console.log('Set Projection to :' ,!Projection)
		// Update upper image (opposite of current image)
		if(!Projection){
			const unique_suffix = "?t=" + new Date().getTime();
			setCurrentUpperImage(selectedProjctionImage + unique_suffix);
		  } else {
			const unique_suffix = "?t=" + new Date().getTime();
			setCurrentUpperImage(selectedSegmentationImage + unique_suffix);
		  }

		setProjection(!Projection);


		const color = Projection ? 'green' : 'red';
		const text = Projection ? 'ON' : 'OFF';
	
		// Send dataToSend to backend
	
		// Change the button text and color
		const buttonElement = document.getElementById('toggleButton');
		if (buttonElement) {
			buttonElement.style.backgroundColor = color;
			buttonElement.innerHTML = text;
		}



		
	}

	
	const handleImageClick = (event: MouseEvent) => {

		//////////////////Display the 2D coordinates of the clicked point///////////////////
		// Get the coordinates of the click on the image
		let x_norm = null;
		let y_norm = null;
		let color = 'lightgreen';
		let x = null;
		let y = null;

		if(showImageHelper){
			let left = 0;
			let top = 0;
			let width = 0;
			let height = 0;
			if (imgRef.current) {
				const scale = Math.min(imgRef.current.clientWidth / imgRef.current.naturalWidth, imgRef.current.clientHeight / imgRef.current.naturalHeight);
				width = imgRef.current.naturalWidth * scale;
				height = imgRef.current.naturalHeight * scale;
				left = (imgRef.current.clientWidth - width) / 2;
				top = (imgRef.current.clientHeight - height) / 2;
			}
			//console.log(`clientWidth, clientHeight: (${imgRef.current?.clientWidth}, ${imgRef.current?.clientHeight})`);
			//console.log(`naturalWidth, naturalHeight: (${imgRef.current?.naturalWidth}, ${imgRef.current?.naturalHeight})`);
			//console.log(`left, top: (${left}, ${top})`);

			x = event.offsetX - left;
			y = event.offsetY - top;
			
			//console.log(`Image clicked at coordinates (${x}, ${y})`);

			
			// Rescale the coordinates to 0-1 range
			const target = event.currentTarget as HTMLImageElement;
			x_norm = x / width
			y_norm = y / height
			//console.log(`Height, Width: (${target.height}, ${target.width})`);
			//console.log(`Normalized coordinates: (${x_norm}, ${y_norm})`);
		}else{
			x = event.offsetX;
			y = event.offsetY;
			//console.log(`Image clicked at coordinates (${x}, ${y})`);
			//console.log(`event`)
			//console.log(event)
			const target = event.target as HTMLDivElement;
			x_norm = x / target.clientWidth
			y_norm = y / target.clientHeight
			//console.log(`Height, Width: (${target.clientHeight}, ${target.clientWidth})`);
			//console.log(`Normalized coordinates: (${x_norm}, ${y_norm})`);

		}
		const coord: Coordinate = { x: x_norm, y: y_norm };


		if (Projection){
			// Add the coordinates to the list
			setCurrentVideoPath(originalImagePath)
			setCoordLs2D([...CoordLs2D, coord]);
			//console.log("CoordLs2D: " );
			//console.log(CoordLs2D);
			//console.log("currentImage: " );
			//console.log(currentVideoPath);

			
		
		}else{

			if(mode === 'Add'){
				setPositiveKeypoints([...positiveKeypoints, coord]);
				//console.log("positiveKeypoints: " );
				//console.log(positiveKeypoints);

			}else{
				setNegativeKeypoints([...negativeKeypoints, coord]);
				//console.log("negativeKeypoints: " );
				//console.log(negativeKeypoints);
				// Set color to red
				color = 'red';
			}

		}

			// Create a new div
			const pointDiv = document.createElement("div");
			// Style the div to look like a dot
			pointDiv.style.position = "absolute";
			pointDiv.style.top = (y - 5) + "px"; // subtracting half the size to center the dot
			pointDiv.style.left = (x - 5) + "px"; // subtracting half the size to center the dot
			pointDiv.style.width = "10px";
			pointDiv.style.height = "10px";
			pointDiv.style.backgroundColor = color;
			pointDiv.style.borderRadius = "50%";
		
			// Append the div to the image container
			const imageContainer = document.getElementById("image-container");
			if (imageContainer) {
				imageContainer.appendChild(pointDiv);
			} else {
				console.error("Image container not found");
			}

	};

	const handleCleanKeepPoints = () => {

		// Remove all dots from the image container, except the first one
		const imageContainer = document.getElementById("image-container");
		if (imageContainer) {
			while (imageContainer.children.length > 1) {
				imageContainer.removeChild(imageContainer.lastChild as Node);
			}
		} else {
			console.error("Image container not found");
		}

	
	  };
	const handleClean = () => {


		setPositiveKeypoints([]);
		setNegativeKeypoints([]);
		setPositiveBrushStrokes([]);
		setNegativeBrushStrokes([]);
		setCoordLs2D([]);
		setConfidences([]);
		// Remove all dots from the image container, except the first one
		const imageContainer = document.getElementById("image-container");
		if (imageContainer) {
			while (imageContainer.children.length > 1) {
				imageContainer.removeChild(imageContainer.lastChild as Node);
			}
		} else {
			console.error("Image container not found");
		}

	
	  };
	const handleClean3D = () => {
		setIsClean3DClicked(true);

		//console.log('Button Clean 3D clicked in App.tsx');

		setCoordinates3D([]);
	};
	
	const handleUndo = () => {
		if(brushMode){
			if(mode === 'Add'){
				setPositiveBrushStrokes(PositiveBrushStrokes.slice(0, -1));
			}else{
				setNegativeBrushStrokes(NegativeBrushStrokes.slice(0, -1));
			}
			drawPoints();


		}else{
		if (Projection){
			// Remove the last element from the list
			setCoordLs2D(CoordLs2D.slice(0, -1));
		}else{
			if(mode === 'Add'){
				// Remove the last element from the list
				positiveKeypoints.pop();

			}else{
				// Remove the last element from the list
				negativeKeypoints.pop();
				
			}
		}
		// Remove the last dot from the image container
		drawPoints();
		if(AUTO_PROCESS_COORDINATE){
		handleProcessCoordinates();
		}
	}
	};



	async function handleProcessCoordinatesWithPrevMask() {
		
		// Load the waiting GIF
		const imageContainer = document.getElementById('image-container-2');
		const upperImage = document.getElementById('upper-image');
		const loader = document.getElementById('loader-upper');
		const upperCanvasImg = document.getElementById('upper-canvas-img');
	
		if (imageContainer) {
			if (upperImage) {
				upperImage.style.display = 'none';
			}
			if (upperCanvasImg) {
				upperCanvasImg.style.display = 'none';
			}
			if (loader) {
				loader.style.display = 'block';
			}
		}
	
		setDisableImageClick(true);
		//console.log("handleProcessCoordinates called");
		const videoElement = videoRef.current;
		if (!videoElement) return;
	
		//console.log("videoElement.currentTime: ", videoElement.currentTime);
	
		if (Projection) { // Pass both 2D and 3D coordinates
			if (!canvasImage) return;
			try {
				const response = await fetch('http://' + host + ':' + port + '/pose-estimation', {
					method: 'POST',
					headers: {
						'Content-Type': 'application/json'
					},
					body: JSON.stringify({
						"user": user,
						"2d-coordinates": CoordLs2D,
						"3d-coordinates": coordinates3D,
						"Category": selectedCategory,
						"SubCategory": selectedSubCategory,
						"Object": selectedStep,
						'currentModelFilePaths': currentModelFilePaths
					})
				});
				
				const data = await response.json();
				processData(data);
				setHasNewAnnotation(true)
	
			} catch (error) {
				if (error instanceof Error) {
					if (error.message.includes('ERR_CONNECTION_REFUSED') || error.message.includes('Network request failed') || error.message.includes('Failed to fetch')) {
						// Here you can handle the specific error
						// For example, by refreshing the page:
						console.log('Connection refused. Refreshing page...');
						window.location.reload();
					}
				}
				console.error("An error occurred:", error);
			}
	
		} else { // Pass only positive and negative keypoints, and display the segmentation result at the original image
			if (!canvasImage) return;
			console.log('video', selectedVideo)
			try {
				const response = await fetch('http://' + host + ':' + port + '/segmentation', {
					method: 'POST',
					headers: {
						'Content-Type': 'application/json'
					},
					body: JSON.stringify({
						"user": user,
						"positive-keypoints": positiveKeypoints,
						"negative-keypoints": negativeKeypoints,
						"Category": selectedCategory,
						"SubCategory": selectedSubCategory,
						"Object": selectedStep,
						'currentModelFilePaths': currentModelFilePaths,
						'PositiveBrushStrokes': PositiveBrushStrokes,
						'NegativeBrushStrokes': NegativeBrushStrokes,
						'with_prev_mask': true,

						"step": selectedStep,
						"video": selectedVideo,
						"part": currentModelFilePaths,
						"frame": selectedFrame,
					})
				});
	
				const data = await response.json();
				processData(data);
				setHasNewAnnotation(true)
	
			} catch (error) {
				if (error instanceof Error) {
					if (error.message.includes('ERR_CONNECTION_REFUSED') || error.message.includes('Network request failed') || error.message.includes('Failed to fetch')) {
						// Here you can handle the specific error
						// For example, by refreshing the page:
						console.log('Connection refused. Refreshing page...');
						window.location.reload();
					}
				}
				console.error("An error occurred:", error);
			}
		}
	
		async function processData(data: any) {
			if (upperImage) {
				upperImage.style.display = 'block';
			}
			if (loader) {
				loader.style.display = 'none';
			}
			if (upperCanvasImg) {
				upperCanvasImg.style.display = 'block';
			}
			setDisableImageClick(false);
	
			if (data.error) {
				handleDataError(data.error);
			} else {
				if (data.messages) {
					setMessages(data.messages);
				}
				if (Projection) {
					const unique_suffix = "?t=" + new Date().getTime();
					setCurrentUpperImage(data.imagePath[0] + unique_suffix);
					setImages(data.imagePath);	
					console.log('data.imagePath',data.imagePath)
				}else{
				const unique_suffix = "?t=" + new Date().getTime();
				setCurrentUpperImage(data.imagePath + unique_suffix);
				}
			}
		}
	
		async function handleDataError(error: any) {
			//console.log("Error received:", error);
			if (error === "No part selected. Please select at least one part for segmentation.") {
				alert("No part selected. Please select at least one part for segmentation.");
			} else if (error === 'SolvePnP failed.') {
				alert('SolvePnP failed. Please check if the selected points are correct.');
			} else if (error === 'Not enough keypoints. Please select at least 4 keypoints.') {
				alert('Not enough keypoints. Please select at least 4 keypoints.');
			} else if (error === 'Keypoints shape mismatch.') {
				alert('Keypoints shape mismatch. Please check if the selected points are correct.');
			} else {
				setMessages(error);
				setCurrentUpperImage(''); // Clear the upper image
			}
		}
	}
	async function handleProcessCoordinates() {
		
		// Load the waiting GIF
		const imageContainer = document.getElementById('image-container-2');
		const upperImage = document.getElementById('upper-image');
		const loader = document.getElementById('loader-upper');
		const upperCanvasImg = document.getElementById('upper-canvas-img');
	
		if (imageContainer) {
			if (upperImage) {
				upperImage.style.display = 'none';
			}
			if (upperCanvasImg) {
				upperCanvasImg.style.display = 'none';
			}
			if (loader) {
				loader.style.display = 'block';
			}
		}
	
		setDisableImageClick(true);
		//console.log("handleProcessCoordinates called");
		const videoElement = videoRef.current;
		if (!videoElement) return;
	
		//console.log("videoElement.currentTime: ", videoElement.currentTime);
	
		if (Projection) { // Pass both 2D and 3D coordinates
			if (!canvasImage) return;
			try {
				const response = await fetch('http://' + host + ':' + port + '/pose-estimation', {
					method: 'POST',
					headers: {
						'Content-Type': 'application/json'
					},
					body: JSON.stringify({
						"user": user,
						"2d-coordinates": CoordLs2D,
						"3d-coordinates": coordinates3D,
						"Category": selectedCategory,
						"SubCategory": selectedSubCategory,
						"Object": selectedStep,
						'currentModelFilePaths': currentModelFilePaths
					})
				});
				
				const data = await response.json();
				processData(data);
				setHasNewAnnotation(true)
	
			} catch (error) {
				if (error instanceof Error) {
					if (error.message.includes('ERR_CONNECTION_REFUSED') || error.message.includes('Network request failed') || error.message.includes('Failed to fetch')) {
						// Here you can handle the specific error
						// For example, by refreshing the page:
						console.log('Connection refused. Refreshing page...');
						window.location.reload();
					}
				}
				console.error("An error occurred:", error);
			}
	
		} else { // Pass only positive and negative keypoints, and display the segmentation result at the original image
			if (!canvasImage) return;
			try {
				const response = await fetch('http://' + host + ':' + port + '/segmentation', {
					method: 'POST',
					headers: {
						'Content-Type': 'application/json'
					},
					body: JSON.stringify({
						"user": user,
						"positive-keypoints": positiveKeypoints,
						"negative-keypoints": negativeKeypoints,
						"Category": selectedCategory,
						"SubCategory": selectedSubCategory,
						"Object": selectedStep,
						'currentModelFilePaths': currentModelFilePaths,
						'PositiveBrushStrokes': PositiveBrushStrokes,
						'NegativeBrushStrokes': NegativeBrushStrokes,
						'with_prev_mask': false

					})
				});
	
				const data = await response.json();
				processData(data);
				setHasNewAnnotation(true)
	
			} catch (error) {
				if (error instanceof Error) {
					if (error.message.includes('ERR_CONNECTION_REFUSED') || error.message.includes('Network request failed') || error.message.includes('Failed to fetch')) {
						// Here you can handle the specific error
						// For example, by refreshing the page:
						console.log('Connection refused. Refreshing page...');
						window.location.reload();
					}
				}
				console.error("An error occurred:", error);
			}
		}
	
		async function processData(data: any) {
			if (upperImage) {
				upperImage.style.display = 'block';
			}
			if (loader) {
				loader.style.display = 'none';
			}
			if (upperCanvasImg) {
				upperCanvasImg.style.display = 'block';
			}
			setDisableImageClick(false);
	
			if (data.error) {
				handleDataError(data.error);
			} else {
				if (data.messages) {
					setMessages(data.messages);
				}
				if (Projection) {
					const unique_suffix = "?t=" + new Date().getTime();
					setCurrentUpperImage(data.imagePath[0] + unique_suffix);
					setImages(data.imagePath);	
					console.log('data.imagePath',data.imagePath)
				}else{
				const unique_suffix = "?t=" + new Date().getTime();
				setCurrentUpperImage(data.imagePath + unique_suffix);
				}
			}
		}
	
		async function handleDataError(error: any) {
			//console.log("Error received:", error);
			if (error === "No part selected. Please select at least one part for segmentation.") {
				alert("No part selected. Please select at least one part for segmentation.");
			} else if (error === 'SolvePnP failed.') {
				alert('SolvePnP failed. Please check if the selected points are correct.');
			} else if (error === 'Not enough keypoints. Please select at least 4 keypoints.') {
				alert('Not enough keypoints. Please select at least 4 keypoints.');
			} else if (error === 'Keypoints shape mismatch.') {
				alert('Keypoints shape mismatch. Please check if the selected points are correct.');
			} else {
				setMessages(error);
				setCurrentUpperImage(''); // Clear the upper image
			}
		}
	}
	

	

	
	const handle3DCoordinates = (newCoordinates: THREE.Vector3) => {
		//console.log({"coordinates3D before update: ": coordinates3D});
		setCoordinates3D([...coordinates3D, newCoordinates]);


		//console.log({"coordinates3D after update: ": coordinates3D});
	}

	// Handle save mask button click
const handleSaveMask = async () => {
	// if (annotationForCurrentFrameExist){
	// 	const r = window.confirm("Previous annotation for this frame exists. Do you want to overwrite it?");
	// 	if (r === false) {
	// 		return ;
	// 	}
	// }
    //console.log("Save Mask called");
    const videoElement = videoRef.current;
    if (!videoElement) return;
    //console.log("videoElement.currentTime: ", videoElement.currentTime);

    try {
        // Send save mask request to backend
        const response = await fetch('http://'+host + ':' + port + '/save-mask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
				"user": user,
				"category": selectedCategory,
				"name": selectedSubCategory,
				"step": selectedStep,
				"video": selectedVideo,
				"part": currentModelFilePaths,
				"frame": selectedFrame,

			 })
        });

        const data = await response.json();
        //console.log(data);

        // Handle response data here
        if (data.error) {
            //console.log("Error received:", data);
            alert(data.error);
        }
        if (data.messages) {
            setMessages(data.messages);
            // alert(data.messages);
        }
		const unique_suffix = "?t=" + new Date().getTime();
		setCurrentUpperImage(data.imagePath + unique_suffix);	

    } catch (error) {
		if (error instanceof Error) {
			if (error.message.includes('ERR_CONNECTION_REFUSED') || error.message.includes('Network request failed') || error.message.includes('Failed to fetch')) {
				// Here you can handle the specific error
				// For example, by refreshing the page:
				console.log('Connection refused. Refreshing page...');
				window.location.reload();
			}
		}
        console.error('Error:', error);
    }
	updateProgressBar()
}


	// Set category at the web load
	useEffect(() => {
		fetch('http://'+host + ':' + port + '/get-categories', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json'
			},
			body: JSON.stringify({ "user": user})
		})
			.then(response => response.json())
			.then(data => {
				if (data.error){
					//console.log("Error received:", data);
					alert(data.error);
				}
				if (data.messages){
					setMessages(data.messages);
				}
				//console.log(data);
				setCategories(data.categories);
			})
			
			.catch(error => {
				console.error('Error making POST request:', error);
				if (error.message.includes('ERR_CONNECTION_REFUSED') || error.message.includes('Failed to fetch')) {
				  console.log('Connection refused. Refreshing page...');
				  window.location.reload();
				}
			})

			if (videoRef.current) {
				videoRef.current.currentTime = 0;
			}
			
		}, []);


	const handleCategoryChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
		setSelectedCategory(event.target.value);
		setSelectedSubCategory("");
		setSelectedStep("");
		setSubCategories([]);
		setSteps([]);
		setVideos([]);

		if (event.target.value !== "") {
			fetch('http://'+host + ':' + port + '/get-subcategories', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({  "user": user,  category: event.target.value})
			})
			.then(response => response.json())
			.then(data => {
				if (data.error){
					//console.log("Error received:", data);
					alert(data.error);
				}
				if (data.messages){
					setMessages(data.messages);
				}
				//console.log(data);
				setSubCategories(data.subcategories);

			}
			)
			.catch(error => {
				console.error('Error making POST request:', error);
				if (error.message.includes('ERR_CONNECTION_REFUSED') || error.message.includes('Failed to fetch')) {
				  console.log('Connection refused. Refreshing page...');
				  window.location.reload();
				}
			})
		}
	}

	const handleSubCategoryChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
		setMessages([]);
		setSelectedSubCategory(event.target.value);
		setSelectedStep("");
		setSteps([]);
		

		if (event.target.value !== "") {
			fetch('http://'+host + ':' + port + '/get-steps', {

				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({  "user": user,  category: selectedCategory, subCategory: event.target.value})
			})
			.then(response => response.json())
			.then(data => {
				if (data.error){
					//console.log("Error received:", data);
					alert(data.error);
				}
				if (data.messages){
					setMessages(data.messages);
				}
				//console.log(data);
				setSteps(data.steps);
				setAllModelFilePaths(data.filePaths);

				const initialColors: Record<string, string> = {};
				data.filePaths.forEach((path: string) => {
				  initialColors[path] = '#D3D3D3';
										});			
				setHighlightColors(initialColors);
				//console.log('initialColors: ',initialColors)
			}
			)
			.catch(error => {
				console.error('Error making POST request:', error);
				if (error.message.includes('ERR_CONNECTION_REFUSED') || error.message.includes('Failed to fetch')) {
				  console.log('Connection refused. Refreshing page...');
				  window.location.reload();
				}
			})
		}
	}

	const handleStepChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
		setMessages([]);
		setSelectedStep(event.target.value);
		if (event.target.value !== "") {
			fetch('http://'+host + ':' + port + '/get-video-manual-parts', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({  "user": user,  category: selectedCategory, subCategory: selectedSubCategory, step: event.target.value })
			})
			.then(response => response.json())
			.then(data => {
				if (data.error){
					//console.log("Error received:", data);
					alert(data.error);
				}
				if (data.messages){
					setMessages(data.messages);
				}
				//console.log(data);
				// if (data.length > 0) {
				  setVideos(data.originalVideoPaths);
				  //console.log("original image paths: ");
				  //console.log(data.originalVideoPaths);
				  setManualImagePath(data.manualImagePath);
				  //console.log("manual image path: ");
				  //console.log(data.manualImagePath);
				  
				  
			  })
			
			  .catch(error => {
				console.error('Error making POST request:', error);
				if (error.message.includes('ERR_CONNECTION_REFUSED') || error.message.includes('Failed to fetch')) {
				  console.log('Connection refused. Refreshing page...');
				  window.location.reload();
				}
			})
		}
	}
	const [allFrames, setAllFrames] = useState<string[]>([]);

	useEffect(() => {
		if (videos.length > 0) {
			fetch('http://'+host + ':' + port + '/get-frames' , {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
	
				body: JSON.stringify({  "user": user, 'Video': selectedVideo, "Category": selectedCategory, "SubCategory": selectedSubCategory, "Step": selectedStep, 'Projection': Projection})
			})
			.then(response => response.json())
			.then(data => {
				if (data.error){
					console.log("Error received:", data);
					// alert(data.error);
				}
				if (data.messages){
					setMessages(data.messages);
				}
				//console.log(data);
				// if (data.length > 0) {
				setAllFrames(data.frames);
				console.log("all frames: ", data.frames);
				  
			  })
			
			  .catch(error => {
				console.error('Error making POST request:', error);
				console.log(error.message)
				if (error.message.includes('ERR_CONNECTION_REFUSED') || error.message.includes('Failed to fetch')) {
				  console.log('Connection refused. Refreshing page...');
				  window.location.reload();
				}
			})

		}
	}, [Projection]); 

	const handleVideoChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
		//console.log("HandleVideoChange Called")
		setMessages([]);
		//reinitilize
		setNextFrameTime(0)
		setPositiveKeypoints([]);
		setNegativeKeypoints([]);
		setCoordLs2D([]);
		setConfidences([]);
		handleClean();
		setSelectedPointIndex(null);
		setIsDragging(false);
		setPointsInitilized(false);
		setAnnotationForCurrentFrameExist(false)
		setHasNewAnnotation(false)
		

		// Get model file 
		fetch('http://'+host + ':' + port + '/get-frames' , {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json'
			},

			body: JSON.stringify({  "user": user, 'Video': event.target.value, "Category": selectedCategory, "SubCategory": selectedSubCategory, "Step": selectedStep, 'Projection': Projection})
		})
		.then(response => response.json())
		.then(data => {
			if (data.error){
				//console.log("Error received:", data);
				alert(data.error);
			}
			if (data.messages){
				setMessages(data.messages);
			}
			//console.log(data);
			// if (data.length > 0) {
			setAllFrames(data.frames);
			console.log("all frames: ", data.frames);
			  
		  })
		
		  .catch(error => {
			console.error('Error making POST request:', error);
			console.log(error.message)
			if (error.message.includes('ERR_CONNECTION_REFUSED') || error.message.includes('Failed to fetch')) {
			  console.log('Connection refused. Refreshing page...');
			  window.location.reload();
			}
		})

		

		setSelectedVideo(event.target.value); // get the last element of the path
		setCurrentVideoPath(event.target.value);
		//console.log("current image: ");
		//console.log(event.target.value);

		// Clean canvas and upper image
		setCurrentUpperImage('');
		//console.log("canvasRef")
		//console.log(canvasRef)
		//console.log(canvasRef.current)
		const canvasElement = canvasRef.current;
		if (canvasElement) {
			//console.log("CanvasElement exist")
			const context = canvasElement.getContext('2d');
			if (context) {
				//console.log("content exist")
				context.clearRect(0, 0, canvasElement.width, canvasElement.height);
				//console.log("Remove content")
			}
		}else{
			//console.log("CanvasElement not exist")
			return;
		}


		if (imgRef.current) {
			//console.log(imgRef.current.naturalWidth);
			//console.log(imgRef.current.naturalHeight);
			//console.log(imgRef.current.clientWidth);
			//console.log(imgRef.current.clientHeight);
		  }

		setOriginalImagePath(event.target.value);





	}
	const [selectedFrame, setSelectedFrame] = useState<string>('');
	const handleFrameChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
		//console.log("HandleVideoChange Called")
		setMessages([]);
		//reinitilize
		setNextFrameTime(0)
		// setPositiveKeypoints([]);
		// setNegativeKeypoints([]);
		setCoordLs2D([]);
		setConfidences([]);
		// handleClean();
		setSelectedPointIndex(null);
		setIsDragging(false);
		setPointsInitilized(false);
		setAnnotationForCurrentFrameExist(false)
		setHasNewAnnotation(false)
		setSelectedFrame(event.target.value);
		setCurrentModelFilePaths([]);

		// Get model file 
		fetch('http://'+host + ':' + port + '/get-model-path-lists' , {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json'
			},

			body: JSON.stringify({ "user": user, 'Video': selectedVideo, "Category": selectedCategory, "SubCategory": selectedSubCategory, "Step": selectedStep, "Frame": event.target.value})
		})
		.then(response => response.json())
		.then(data => {
			if (data.error){
				//console.log("Error received:", data);
				alert(data.error);
			}
			if (data.messages){
				setMessages(data.messages);
			}

			setModelFilePathLists(data.modelFilePathLists);
			setCurrentFrameTime(data.currentFrameTime);
			  
		  })
		
		  .catch(error => {
			console.error('Error making POST request:', error);
			console.log(error.message)
			if (error.message.includes('ERR_CONNECTION_REFUSED') || error.message.includes('Failed to fetch')) {
			  console.log('Connection refused. Refreshing page...');
			  window.location.reload();
			}
		})
		

		

	}

	useEffect(() => {
		setAllModelFilePaths([]);
		

		fetch('http://'+host + ':' + port + '/get-steps', {

		method: 'POST',
		headers: {
			'Content-Type': 'application/json'
		},
		body: JSON.stringify({  "user": user,  category: selectedCategory, subCategory: selectedSubCategory})
	})
	.then(response => response.json())
	.then(data => {
		if (data.error){
			//console.log("Error received:", data);
			alert(data.error);
		}
		if (data.messages){
			setMessages(data.messages);
		}
		//console.log(data);
		// setSteps(data.steps);
		setAllModelFilePaths(data.filePaths);

		
		const newHighlightColors: Record<string, string> = {};
		data.filePaths.forEach((path: string) => {
			if (currentModelFilePaths.includes(path)){
				newHighlightColors[path] = '#90C3D4';
			}else{
				newHighlightColors[path] = '#D3D3D3';
			}
								});			
		setHighlightColors(newHighlightColors);
		//console.log('initialColors: ',initialColors)
	}
	)
	.catch(error => {
		console.error('Error making POST request:', error);
		if (error.message.includes('ERR_CONNECTION_REFUSED') || error.message.includes('Failed to fetch')) {
		  console.log('Connection refused. Refreshing page...');
		  window.location.reload();
		}
	})
		
	}, [showImageHelper2])

	const handleObjectListChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
		setNextFrameTime(0)
		// setPositiveKeypoints([]);
		// setNegativeKeypoints([]);
		// setCoordLs2D([]);
		setConfidences([]);
		// handleClean();
		setSelectedPointIndex(null);
		setIsDragging(false);
		setPointsInitilized(false);
		setAnnotationForCurrentFrameExist(false)
		setHasNewAnnotation(false)

		
		//console.log("HandleObjectListChange Called")
        setCurrentModelFilePaths(event.target.value.split(','));
		// set highlight colors if the model file paths are in the list event.target.value.split(',')
		const newHighlightColors: Record<string, string> = {};
		allModelFilePaths.forEach((path: string) => {
			if (event.target.value.split(',').includes(path)){
				newHighlightColors[path] = '#90C3D4';
			}else{
				newHighlightColors[path] = '#D3D3D3';
			}
		}
		);
		setHighlightColors(newHighlightColors);

		if (event.target.value !== "") {
			console.log('user', user, 'category', selectedCategory, 'subCategory', selectedSubCategory, 'step', selectedStep, 'video', selectedVideo, 'part', event.target.value.split(','), 'frame', selectedFrame);
		  
			fetch('http://' + host + ':' + port + '/get-prev-mask', {
			  method: 'POST',
			  headers: {
				'Content-Type': 'application/json'
			  },
			  body: JSON.stringify({
				user: user,
				category: selectedCategory,
				subCategory: selectedSubCategory,
				step: selectedStep,
				video: selectedVideo,
				part: event.target.value.split(','),
				frame: selectedFrame
			  })
			})
			  .then(response => response.json())
			  .then(data => {
				if (data.error) {
				  alert(data.error);
				}
				if (data.messages) {
				  setMessages(data.messages);
				}
				const base64Image = `data:image/png;base64,${data.image}`;
				setPrevMaskImagePath(base64Image);
				console.log("prev mask image path: ", base64Image);
				const base64ImageOverlay = `data:image/png;base64,${data.image_overlay}`;
				setPrevMaskImageOverlayPath(base64ImageOverlay);
				console.log("prev mask overlay image path: ", base64ImageOverlay);
				setPrevMaskReason(data.reason);
			  });
		  }

		

    };

	const [opacity, setOpacity] = useState(1);

	const handleSaveProjectionBtnClick = async () =>{
		handleSaveProjection(true);
	}

	const handleSaveProjection = async (useShareFrames: boolean) => {
		//console.log("handleProcessCoordinates called");
		const videoElement = videoRef.current;
		if (!videoElement) return;
		//console.log("videoElement.currentTime: ", videoElement.currentTime);
	
		try {
			const response = await fetch(`http://${host}:${port}/save-projection`, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({
					"user": user,
					'Category': selectedCategory,
					'SubCategory': selectedSubCategory,
					'Object': selectedStep,
					'currentModelFilePaths': currentModelFilePaths,
					"category": selectedCategory,
					"name": selectedSubCategory,
					"step": selectedStep,
					"video": selectedVideo,
					"part": currentModelFilePaths,
					"shareFrames": selectedShareFrames,
					"useShareFrames": useShareFrames
				})
			});
	
			const data = await response.json();
			//console.log(data);
	
			if (data.error) {
				//console.log("Error received:", data);
				alert(data.error);
			}
			if (data.messages) {
				setMessages(data.messages);
			}
			const unique_suffix = "?t=" + new Date().getTime();
			setCurrentUpperImage(data.imagePath + unique_suffix);
		} catch (error) {
			if (error instanceof Error) {
				if (error.message.includes('ERR_CONNECTION_REFUSED') || error.message.includes('Network request failed') || error.message.includes('Failed to fetch')) {
					// Here you can handle the specific error
					// For example, by refreshing the page:
					console.log('Connection refused. Refreshing page...');
					window.location.reload();
				}
			}
			console.error('Error:', error);
		}
		updateProgressBar()
	}

	useEffect(() => {
		const canvasElement = canvasRef.current;
		if (canvasElement) {
		  const newImageSrc = canvasElement.toDataURL("image/png");
		  setCanvasImageSrc(newImageSrc);
		}
		//console.log("canvasImageSrc: ", canvasImageSrc);
	  }, [canvasRef]);

	const handleProjctionImage = (event: React.ChangeEvent<HTMLSelectElement>) => {
		setSelectedProjectionImage(event.target.value);
		//console.log("selected projection image: ");
		//console.log(event.target.value);
		const unique_suffix = "?t=" + new Date().getTime();
		setCurrentUpperImage(event.target.value + unique_suffix);
	}

	const handleSegmentationImage = (event: React.ChangeEvent<HTMLSelectElement>) => {
		setSelectedSegmentationImage(event.target.value);
		//console.log("selected projection image: ");
		//console.log(event.target.value);
		const unique_suffix = "?t=" + new Date().getTime();
		setCurrentUpperImage(event.target.value + unique_suffix);
	}

	const handleInput1Change = (e : React.ChangeEvent<HTMLInputElement>) => {
		setInput1(e.target.value);
	  };
	
	  const handleInput2Change = (e : React.ChangeEvent<HTMLInputElement>) => {
		setInput2(e.target.value);
	  };
	
	  const handleInput3Change = (e : React.ChangeEvent<HTMLInputElement>) => {
		setInput3(e.target.value);
	  };
	
	  const handleSubmit = async () => {
		// setMessages([]);
		fetch('http://'+host + ':' + port + '/change-current-projection', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json'
			},
			body: JSON.stringify({  "user": user, 'Input1': input1, 'Input2': input2, 'Input3': input3})
		})
		.then(response => response.json())
		.then(data => {
			if (data.error){
				//console.log("Error received:", data);
				alert(data.error);
			}
			if (data.messages){
				setMessages(data.messages);
			}
			//console.log(data);
		}
		)
		.catch(error => {
			console.error('Error making POST request:', error);
			if (error.message.includes('ERR_CONNECTION_REFUSED') || error.message.includes('Failed to fetch')) {
			  console.log('Connection refused. Refreshing page...');
			  window.location.reload();
			}
		})


	  };

	
	
	return (
		<main className={props.className}>
			<div className="gallery-container">
				{/* ---------------------- Left Section ----------------- */}
				<div className="left-section">
					
					
					<select value={selectedCategory} onChange={handleCategoryChange}>
						<option value="">Select Category</option>
						{/* List categories here */}
						{Categories.map(category => (
							<option value={category}>{category}</option>
						))}
						</select>
					<br />
					<select value={selectedSubCategory} onChange={handleSubCategoryChange}>
						<option value="">Select Sub Category</option>
						{subCategories && subCategories.map(subCategory => (
							<option value={subCategory}>{subCategory}</option>
						))}
					</select>

					<br />
					<select value={selectedStep} onChange={handleStepChange}>
						<option value="">Select Step</option>
						{/* List objects here */}
						{steps.map(object => (
							<option value={object}>{object}</option>
						))}
					</select>

					<br />
					<select value={selectedVideo} onChange={handleVideoChange}>
					<option value="">Select Video</option>
					{/* List videos here */}
					{videos && videos.map((video: string) => {
						const fileName = video.split('/').pop(); // Extract the file name from the URL
						return (
						<option value={video}>{fileName}</option>
						);
					})}
					</select>

					<br />
					<div>
					<select 
						value={selectedFrame}
						onChange={handleFrameChange}
					>
						<option value="">Select Frame</option>
						{allFrames &&( allFrames.map((list, index) => (
							
							<option key={index} value={list}>
								{list}
							</option>
							
						)))}
					</select>

						<br />
					</div>
					<br />
					
					<div>
					<select 
						value={currentModelFilePaths} 
						onChange={handleObjectListChange}
						disabled={nextFrameTime !== 0}  // disable the select input when nextFrameTime is not 0
					>
						<option value="">Select Object</option>
						{modelFilePathLists &&( modelFilePathLists.map((list, index) => (
							<option key={index} value={list}>
								{list?.map(item => item?.split('/').pop()?.split('.')[0]).join(",")}
							</option>
						)))}
					</select>

						<br />
					</div>
					<br />

					<SwitchModeButton  id= "projection-segmentation-selection" onClick={handleProjectionToggleButtonClick}>
						{Projection ? "Current Mode: Projection" : "Current Mode : Segmentation"}
					</SwitchModeButton>
					<br />

					
					{/* Display manual image */}
					{(ManualImagePath &&
					<img src={ManualImagePath} alt="Manual" />
					)}
					<div style={{ display: Projection ? 'none' : 'block' }}>
					<span>Previous Mask Image: </span>
					</div>
					{PrevMaskImagePath && <img src={PrevMaskImagePath} alt="PrevMask" />}
					<span>Previous Mask Image Overlay: </span>
					{PrevMaskImageOverlayPath && <img src={PrevMaskImageOverlayPath} alt="PrevMask" />}
					<span>Wrong reason: Wrong object/mask incomplete/mask contains other objects</span>
					
					
					
					{selectedVideo && 						
					<video ref={videoRef} id="video_preview" width="100%" height="100%" crossOrigin="anonymous" controls autoPlay loop muted >
						<source src={selectedVideo} type="video/mp4" />
					</video>
					}
							




				</div>
				
				{/* ---------------------- Middle Section ----------------- */}
				<div className="middle-section">
						
					<br />
					<br />
					{/* Displaying progress as a fraction */}
					{nextFrameTime && (
						<div>
							<h2>Don't forget to SAVE changes before leaving!!</h2>

							<div>
								Current subassembly progress: {annotatedCurrPartCurrStep}/{totalCurrPartCurrStep}
							</div> 

							<div>
								{selectedStep} progress: {annotatedCurrStep}/{totalCurrStep}
							</div>

							<div>
								Video {currentVideoPath.split('/').pop()} progress: {annotatedCurrVideo}/{totalCurrVideo}
							</div> 

							<div>
								Current step video {currentVideoPath.split('/').pop()} progress: {annotatedCurrStepCurrVideo}/{totalCurrStepCurrVideo}
							</div> 

							<div>
								Furniture {selectedCategory} - {selectedSubCategory} progress: {annotatedCurrFurniture}/{totalCurrFurniture}
							</div>

							<div>
								Overall progress: {annotatedAllFurniture}/{totalAllFurniture}
							</div> 
						</div>
					)}


					<div>
						{Array.isArray(messages) && messages.map((message, index) => (
							<p key={index} style={{
								fontFamily: 'Arial, sans-serif',
								color: '#333',
								lineHeight: '1.6',
								textAlign: 'left'
							}}>
								{message}
							</p>
						))}
					</div>



	
					<div className="scene-container" style={{ position: 'relative', width: '100%', height: '100%' }}>
						<Scene handle3DCoordinates={handle3DCoordinates} model={<ModelLoader filePaths={showImageHelper2 ? currentModelFilePaths : allModelFilePaths} highlightColors={highlightColors} />} mode= {Projection ? "Projection" : "Segmentation"}
						isClean3DClicked={isClean3DClicked} // Pass the isCleanClicked state as prop
						setIsClean3DClicked={setIsClean3DClicked} // Pass the setIsCleanClicked setter async function as prop
						handleImageClick={handleImageClick}
						showImageHelper={showImageHelper}
						/>
					
					
					<div style={{ display: showImageHelper ? 'block' : 'none', position: 'absolute', width: '100%', height: '100%', pointerEvents: 'none' }}>
						{(canvasImage &&
						<img ref={imgRef} src={canvasImage.src} style={{ width: '100%', height: '100%', objectFit: 'contain', opacity: opacity }} />
						)}

					</div>
					
				</div>
				
				<div className="button-wrapper">

					<ProcessCoordinatesButton className="process-button" id="process-coordinates" onClick={handleProcessCoordinates}>	
						Process Coordinates
					</ProcessCoordinatesButton>
					<ProcessCoordinatesButton className="process-button_prev_mask" id="process-coordinates-prev-mask" onClick={handleProcessCoordinatesWithPrevMask}>
						Process Coordinates with Prev Mask
					</ProcessCoordinatesButton>
					{/* <ProcessCoordinatesButton className="confirm-brush" id="confirm-brush" onClick={handleBrushConfirm}>
						Confirm Brush
					</ProcessCoordinatesButton> */}
					{/* Display save button if in project mode */}

					{/* Points Tracking button */}
					{/* <OtherButton_Green  id="points-tracking-button" onClick={handlePointsTracking}>Points Tracking</OtherButton_Green> */}
					{ POINTS_TRACKING && 
					<OtherButton_Green  id="points-tracking-button" onClick={trackingPointsInit}>Points Initilization</OtherButton_Green>
					}

					

					<div style={{ display: Projection ? 'block' : 'none' }}>
						<OtherButton_Red  id = 'clean-btn-project' onClick={handleClean3D}>Clean 3D Coords</OtherButton_Red>
						{/* switch to control showImageHelper */}

					</div>
					<div style={{ display: Projection ? 'block' : 'none' }}>
					<span>Image Helper</span>
						<input
								type="range"
								min="0"
								max="1"
								value={showImageHelper ? 1 : 0}
								step="1"
								onChange={(event) => setShowImageHelper(event.target.value === "1")}
								style={{ width: '50px' }}
							/>
						</div>
					<div style={{ display: Projection ? 'block' : 'block' }}>
						<span>3D Helper</span>
						<input
								type="range"
								min="0"
								max="1"
								value={showImageHelper2 ? 1 : 0}
								step="1"
								onChange={(event) => setShowImageHelper2(event.target.value === "1")}
								style={{ width: '50px' }}
							/>
						</div>
					</div>
				</div>
				
				{/* ---------------------- Right Section ----------------- */}
				<div className="right-section">

					{/* <Scene setCoordinates3D={setCoordinates3D} model={model[index]} effectOnKeyDown={effectOnKeyDown} /> */}

					<div className="part" id="top-part" style={{ width: '90%', height: '100%' }}>
						
						{/* Place the image of 3d projection */}

 
						<div id="image-container-output" style={{ position: "relative" }}>
							{currentUpperImage && canvasImage && (
							<img id= "upper-canvas-img" src={canvasImage.src} alt="frame_image" onClick={(event) => handleImageClick(event.nativeEvent as any)} />
							)}
							<br />
							<div id="image-container-output" style={{ position: "absolute", top: 0, left: 0 }}>
							{Projection ? (
							<div style={{ overflowX: 'auto', whiteSpace: 'nowrap' }}>
								{/* {images.map((image, index) => (
									<img
										key={index}
										src={image}
										alt={`upper-image-${index}`}
										style={{ display: 'inline', marginRight: '10px', opacity: opacity }}
									/>
								))} */}
								{/* <StyledImageSlider images={images} /> */}
								{currentUpperImage && (
									<img id="upper-image" src={currentUpperImage} alt="upper-image" style={{ opacity: opacity }} />
								)}
							</div>
						) : (
							<div>
								{currentUpperImage && (
									<img id="upper-image" src={currentUpperImage} alt="upper-image" style={{ opacity: opacity }} />
								)}
								
							</div>

						)}
						<br />



						<br />
							<div className='loader' id="loader-upper" style={{ display: "none" }}>
							<div className="circle"></div>
							{Projection ? 'Projection Processing...' : 'Segmentation Processing...'}
							</div>
							
							{/*  Add a slider to control the opacity */}
							</div>
						</div>
					</div>


					
					
	
					<div className="part" id="middle-part" style={{ width: '90%', height: '100%' }}>

						<div id="image-container-2" style={{ position: "relative" }}>
						<canvas
							ref={canvasRef}
							id="canvas"
							width="100%"
							height="100%"
							style={{ position: "absolute", top: 0, left: 0 }}
							onMouseDown={handleMouseDown}
							onMouseMove={handleMouseMove}
							onMouseUp={handleMouseUp}
						>
						</canvas>
						<div className='loader' id="loader-middle" style={{ display: "none" }}>
							<div className="circle"></div>
						Processing...
						</div>
						

						</div>

					</div>

	
					<div className="part" id="bottom-part">
						{/* <!-- Placeholder --> */}
						<br />
						{/* Display the button only when the annotation started */}
						{/* {nextFrameTime && user==='Yunong' && ( */}
						{nextFrameTime && (

							<div style={{ display: Projection ? 'none' : 'block' }}>
							<span>Add</span>
							<input
								type="range"
								min="0"
								max="1"
								value={mode === 'Add' ? 0 : 1}
								onChange={(e) => setMode(e.target.value === '0' ? 'Add' : 'Remove')}
								style={{ width: '50px' }}
							/>
							<span>Remove</span>						
							<OtherButton_Green  id = 'undo-btn' onClick={handleUndo}>Undo</OtherButton_Green>
				
							<OtherButton_Green  id = 'save-btn' onClick={handleSaveMask}>Save Mask</OtherButton_Green>
							<OtherButton_Red id = 'clean-btn' onClick={handleClean}>Clean</OtherButton_Red>
							<br />
							<SwitchModeButton onClick={() => setBrushMode(!brushMode)}>
							{brushMode ? 'Brush Mode' : 'Select Mode'}
							</SwitchModeButton>

							{brushMode && (<span>Brush Size</span>)}	
							{brushMode && (												
							<input
							type="range"
							min="1"
							max="100"
							value={brushSize}
							onChange={(e) => setBrushSize(parseInt(e.target.value))}
							/>
							)}
							{brushMode && (<span>{brushSize}</span>)}	

							</div>
						)}

							
							<br />

							{/* Button for next frame/start annotation */}
							{selectedVideo && (	
							
							<OtherButton_Green onClick={handleNextFrame} style={{ display: nextFrameTime === 0 ? 'block' : 'none' }}>
								{nextFrameTime === 0 ? 'Start Annotation' : ''}
							</OtherButton_Green>
							)}
							{selectedVideo && (nextFrameTime !== 0) && (	
							<OtherButton_Green onClick={handleBack}>{'Back to the last annotation'}</OtherButton_Green>
							)}
							{selectedVideo && (nextFrameTime !== 0) && (	
							<OtherButton_Green onClick={handlePartNotAbleToAnnotate}>{'Obj cannot be annotated'}</OtherButton_Green>
							)}
							{selectedVideo && (nextFrameTime !== 0) && (	
							<OtherButton_Green onClick={handlePartNotShowUp}>{"Obj doesn't appear"}</OtherButton_Green>
							)}
							{selectedVideo && (nextFrameTime !== 0) && (	
							<OtherButton_Green onClick={handleShareAnnotation}>{"Share Annotation"}</OtherButton_Green>
							)}
							
							<Modal isOpen={isModalOpen} onClose={closeModal}>
							{allFrames && allFrames.length > 0 ? (
			
							<div>
							<select onChange={handleSelectChange} multiple value={selectedShareFrames}>
							{allFrames.filter(frame => !selectedFrame.includes(frame)).map((frame, index) => (
								<option key={index} value={frame}>{frame}</option>
							))}
							</select>
							<button onClick={confirmSelection} style={{ marginTop: '10px' }}>Confirm</button>
						  </div>
		  					) : (
							<p>No frames available</p> // Optional: render a fallback or loading state
							)}
								
							</Modal>
						
							{selectedVideo && (nextFrameTime !== 0) && (	
								
								<>

									<br />
									<OtherButton_Green onClick={handleReportMissingComponent}>{'Report Issue about current frame'}</OtherButton_Green>
									<input type="text" value={frameIssueMessage} onChange={handleFrameIssueMessage} />
									<br />

								</>
							)}

							

						{/* TODO: Button for previous frame */}


						{nextFrameTime && (
						<div style={{ display: Projection ? 'block' : 'none' }}>
						<OtherButton_Red  id = 'clean-btn-project-2d' onClick={handleClean}>Clean 2D Coords</OtherButton_Red>
						<OtherButton_Green  id="3d-save-button" onClick={handleSaveProjectionBtnClick}>Save Projection</OtherButton_Green>

						</div>
						)}
							
						
					{nextFrameTime && (
					<div id="opacity-slider">
						<span>Opacity</span>
						<input
						type="range"
						min="0"
						max="1"
						step="0.01"
						value={opacity}
						onChange={(e) => setOpacity(parseFloat(e.target.value))}
						style={{ width: '100px' }}
						/>
					<div style={{ display: Projection ? 'block' : 'none', marginTop: '20px' }}> 
						<span>Scroll to view other views </span>
						<input type="range" min="0" max={images.length - 1} step="1" onChange={handleSliderChange} /> 
					</div>


						<br />
						<br />
						{/* Add 3 text box  to get input from user, and add a button to send all the values to backend, only if projection mode is selected */}
						{/* <div style={{ display: Projection ? 'block' : 'none' }}>

						<div className="App">
							<label> Adjust Current Projection </label>
							<br />
							<label>Select Axis: </label>
							<input type="text" value={input1} onChange={handleInput1Change} />
							<br />
							<label>Enter Angle or Distance: </label>
							<input type="text" value={input2} onChange={handleInput2Change} />
							<br />
							<label>Rotation or Translation (r/t): </label>
							<input type="text" value={input3} onChange={handleInput3Change} />
							<br />
							<OtherButton_Green  id = 'send-btn' onClick={handleSubmit}>Send</OtherButton_Green>
							</div>
							
						</div> */}
						<br />




					</div>
					)}

					</div>
				</div>
			</div>
		</main>
	);
	};

// Add this styling

const StyledRawApp = styled(RawApp)`
    display: flex;
    height: 100vh; // 100% of the viewport height

	.gallery-container {
        display: flex;
        height: 100%;
        width: 100%;
    }

	select {
		/* imporve the appearance for select and input */
		width: 100%;
		// padding: 10px;
		box-sizing: border-box;
		border: 2px solid #ddd;
		transition: border-color 0.3s ease;
	  }
	  input
	  select {
		width: 20%;
		// padding: 10px;
		box-sizing: border-box;
		border: 2px solid #ddd;
		transition: border-color 0.3s ease;
	  }

    input:focus, select:focus {
        border-color: #007BFF;
    }

    .left-section {
        width: 15%;
        background-color: #f5f5f5;
        padding: 20px;
        display: flex;
        flex-direction: column;
    }

    .middle-section {
        flex: 3;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }

    .scene-container {
        width: 100%;
        flex: 3;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .button-wrapper {
        display: flex;
        justify-content: space-around;
        padding: 20px;
    }

    .right-section {
        flex: 3;
        display: flex;
        flex-direction: column;
        background-color: #f5f5f5;
    }

    .projection-segmentation-selection {
        position: absolute;
        bottom: 0;
    }

    .right-section .part {
        flex: 1; // Each part takes equal space
        overflow: auto; // Adds scrollbar if content overflows
    }

    #top-part, #middle-part, #bottom-part {
        // Additional specific styles if needed
    }

    #top-part img, #middle-part img {
        max-width: 100%;
        max-height: 100%;
        object-fit: cover;
    }
	

    #loader {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        z-index: 1;
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.5);
        text-align: center;
        font-size: 18px;
        color: #333;
    }

    .circle {
        width: 24px;
        height: 24px;
        border: 4px solid transparent;
        border-top-color: #333;
        border-radius: 50%;
        animation: spin 2s linear infinite;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
`;


export default StyledRawApp;

