import styled from "styled-components";
import { IStyled } from "../type";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
// import { PerspectiveCamera } from "@react-three/drei";
import { OrbitControls, PerspectiveCamera } from "@react-three/drei";

import { Suspense, useEffect, useRef, useState} from "react";
import * as THREE from "three";
import { TrackballControls } from "@react-three/drei";




interface SceneProps<T = HTMLElement> extends IStyled {
	model?: React.ReactNode;
	handle3DCoordinates: (coordinates: THREE.Vector3) => void; // pdate the SceneProps interface in Scene.tsx to accept the new prop
	// Get current mode from app.tsx (Projection or Segmentation), make sure it is updated
	mode: string;
	isClean3DClicked: boolean; // Add the isCleanClicked prop
  	setIsClean3DClicked: (isClean3DClicked: boolean) => void; // Add the setIsCleanClicked setter function
	handleImageClick: (event: MouseEvent) => void;
	showImageHelper: boolean;


  }
  



const InnerScene = (props: SceneProps<HTMLDivElement>) => {
	// const [coordinates3D_ls, setCoordinates3D_ls] = useState<THREE.Vector3[]>([]);
	const controlsRef = useRef<any>();
	const pointColors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'brown', 'lavender', 'teal', 'tomato', 'gold', 'silver', 'olive', 'maroon', 'navy', 'lime', 'aqua', 'fuchsia', 'black', 'white'];
	const [currentColorIndex, setCurrentColorIndex] = useState(0);
	
	
	const { gl, scene, camera } = useThree();

	useFrame(() => controlsRef.current?.update());

	const raycaster = new THREE.Raycaster();
	const mouse = new THREE.Vector2();
  
	useEffect(() => {
		if (props.isClean3DClicked) {
			// Create an array to hold meshes that need to be removed
			const meshesToRemove: THREE.Object3D[] = [];
	
			// Collect all the meshes
			scene.children.forEach((child) => {
				if (child.type === "Mesh") {
					meshesToRemove.push(child);
				}
			});
	
			// Remove the collected meshes from the scene
			meshesToRemove.forEach((mesh) => {
				scene.remove(mesh);
			});
	
			// Optionally, dispose of the geometry, material, and texture
			meshesToRemove.forEach((mesh) => {
				const currentMesh = mesh as THREE.Mesh;
			
				// Dispose geometry
				currentMesh.geometry.dispose();
			
				// Dispose material(s)
				if (Array.isArray(currentMesh.material)) {
					currentMesh.material.forEach((mat) => mat.dispose());
				} else {
					currentMesh.material.dispose();
				}
			});
			
	
			props.setIsClean3DClicked(false);
			setCurrentColorIndex(0); // reset the color index to 0 after cleaning
		}


	  const handleClick = (event: MouseEvent) => {
		// Make sure the user is in the Projection mode
		if (props.mode === "Segmentation") {
			console.log("Segmentation mode, no 3D coordinates")
			// alert("Segmentation mode, no 3D coordinates")
			return;
		}
		
		console.log("Projection mode, 3D coordinates")
		console.log("event: ", event)
		const rect = gl.domElement.getBoundingClientRect();
		mouse.x = (event.clientX - rect.left) / rect.width * 2 - 1;
		mouse.y = -(event.clientY - rect.top) / rect.height * 2 + 1;
  
		raycaster.setFromCamera(mouse, camera);
  
		const intersects = raycaster.intersectObjects(scene.children, true);
  
		if (intersects.length > 0) {
		  const point = intersects[0].point;
		  console.log('3D Coordinates: ', point);
		  const currentColor = pointColors[currentColorIndex];
		  setCurrentColorIndex((currentColorIndex + 1) % pointColors.length);
		  console.log('Color: ', currentColor);
		  console.log('Color Index: ', currentColorIndex);


			// Display a dot where the user clicked
			const geometry = new THREE.SphereGeometry(0.01);
			// Color the dot with the current color
			const material = new THREE.MeshBasicMaterial({ color: currentColor });
			const sphere = new THREE.Mesh(geometry, material);
			sphere.position.set(point.x, point.y, point.z);
			scene.add(sphere);

			const newCoordinates = point // obtain new 3D coordinates here
			props.handle3DCoordinates(newCoordinates);
			console.log({"New Coord from Scence : ": newCoordinates})

			// Show points on the image if showImageHelper is true
			if (props.showImageHelper){
			props.handleImageClick(event as any);

			}

		}
	  };


		// const perspCamera = camera as THREE.PerspectiveCamera;


  
	  gl.domElement.addEventListener('click', handleClick);
	  return () => gl.domElement.removeEventListener('click', handleClick);
	}, [gl, camera, scene, props.mode, props.handle3DCoordinates, props.isClean3DClicked, props.setIsClean3DClicked]);
  
	return (
		<>
		<ambientLight intensity={0.5} />
		<spotLight intensity={1} position={[5, 5, 5]} />
		{props.showImageHelper && (
			<TrackballControls
				ref={controlsRef}
				args={[camera, gl.domElement]}
				rotateSpeed={5}
				zoomSpeed={15}
				panSpeed={0.8}
				staticMoving={true}
				dynamicDampingFactor={0.3}
			/>)} 
			{!props.showImageHelper && (
			<OrbitControls minDistance={0.1} maxDistance={200} maxPolarAngle={Math.PI} minPolarAngle = {-Math.PI} />
			)}

		<PerspectiveCamera 
			makeDefault 
			position={[10, 10, 5]} 
			near={0.01} 
			far={2000} 
			{...({} as any)}
			/>

		<Suspense fallback={<></>}>{props.model}</Suspense>
	  </>
	);
  };
  

const RawScene = (props: SceneProps<HTMLDivElement>) => {
  return (
    <Canvas tabIndex={0} >
      <InnerScene {...props} />
    </Canvas>
  );
};

const Scene = styled(RawScene)``;

export { Scene };
