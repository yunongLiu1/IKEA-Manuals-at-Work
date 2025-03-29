import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";


let img_width, img_height;
let intrinsics, extrinsics;
let camera, scene, renderer, controls;
let sceneCanvas, imageCanvas, imageContext;
let objPaths = [];
let selectedCategory, selectedName, selectedVideoId, selectedFrameId, selectedUserId;
let category, name, frame_id, video_id, userId;
const port = 5000;
const host = `http://HOST_PLACEHOLDER:${port}`; // e.g. `http://8.15.84.51:${port}`or `http://localhost:${port}`
let transformationLock = false;
const translationScale = 0.01;
const renderResizeRate = 0.7;
let newBBoxToBBox;
let modelMatrix;
let displayAllCoordinates = true;

const displayAllCoordinatesButton = document.getElementById("displayAllCoordinatesButton");
displayAllCoordinatesButton.addEventListener("click", () => {
  displayAllCoordinates = !displayAllCoordinates;
  reloadScene();
});
// let mseScores = [];

let currentRotations = new Proxy({}, {
  set: function (target, key, value) {
    target[key] = value;
    console.log("Current Rotations changed:", target);
    return true;
  }
});

let currentTranslations = new Proxy({}, {
  set: function (target, key, value) {
    target[key] = value;
    console.log("Current Translations changed:", target);
    return true;
  }
});


let currentBBoxToCam = new Proxy({}, {
  set: function (target, key, value) {
    target[key] = value;
    console.log("Current Object Centers to Mesh Center changed:", target);
    return true;
  }
});
let currentMeshToBBox = new Proxy({}, {
  set: function (target, key, value) {
    target[key] = value;
    console.log("Current Mesh To BBox :", target);
    return true;
  }
});





let currentTransfomations = new Proxy({}, {
  set: function (target, key, value) {
    target[key] = value;
    console.log("Current Transfomations changed:", target);
    return true;
  }
});




////////////////// Fetch Camera Parameters and OBJ Paths //////////////////

async function fetchCameraParametersAndObjPaths(selectedCategory, selectedName, selectedVideoId, selectedFrameId) {
  try {
    const response = await fetch(`${host}/camera-parameters-and-obj-paths?category=${selectedCategory}&name=${selectedName}&video_id=${selectedVideoId}&frame_id=${selectedFrameId}&user_id=${selectedUserId}`);
    const data = await response.json();
    console.log('data:', data);
    extrinsics = data.extrinsics;
    intrinsics = data.intrinsics[0];
    img_height = intrinsics[1][2] * 2;
    img_width = intrinsics[0][2] * 2;
    console.log("Camera parameters:", extrinsics, intrinsics);
    console.log("img height:", img_height);
    console.log("img width:", img_width);
    objPaths = data.objPaths;

  } catch (error) {
    console.error("Error fetching camera parameters:", error);

  }
}

// async function fetchObjPaths(selectedCategory, selectedName, selectedVideoId, selectedFrameId) {
//   try {
    
//     const response = await fetch(`${host}/obj-paths?category=${selectedCategory}&name=${selectedName}&video_id=${selectedVideoId}&frame_id=${selectedFrameId}&user_id=${selectedUserId}`);
//     objPaths = await response.json();
//     objPaths = objPaths.objPaths;
//     console.log("OBJ paths:", objPaths);
//   } catch (error) {
//     console.error("Error fetching OBJ paths:", error);
//   }
// }

//////////////////////////// Load Objects ////////////////////////////
async function loadObjects() {
  console.log("Loading objects...", objPaths.length);
  console.log("objPaths:", objPaths);

  
  for (let i = 0; i < objPaths.length; i++) {
    const objectName = 'obj_' + objPaths[i][0].split("/").pop().split(".")[0];
    currentTransfomations[objectName] = new THREE.Matrix4();

    currentBBoxToCam[objectName] = new THREE.Matrix4();
    currentMeshToBBox[objectName] = new THREE.Matrix4();

    let meshes;
    for (let j = 0; j < objPaths[i].length; j++) {
      
      console.log('Loading object:', objPaths[i][j], 'selectedObjectIndex:', selectedObjectIndex, 'i:', i);
      if (selectedObjectIndex !== NaN) {
        if (i === selectedObjectIndex) {
          console.log("Set color to red for object:",objPaths[i][j]);
          const loader = new OBJLoader();
          const object = await loader.loadAsync(objPaths[i][j]);
          object.name = 'obj_' + objPaths[i][j].split("/").pop().split(".")[0];

          object.traverse((child) => {
            if (child instanceof THREE.Mesh) {
              child.material = new THREE.MeshBasicMaterial({
                color: 0xA00000,
                side: THREE.DoubleSide,
              });
            }
          });
          scene.add(object);
          if(!meshes){
            meshes = object;
          }else{
            meshes.add(object);
          }
          continue;
        }
      }

      console.log("Set color to white for object:",objPaths[i][j]);
      const objPath = objPaths[i][j];
      const loader = new OBJLoader();
      const object = await loader.loadAsync(objPath);
      object.name = 'obj_' + objPaths[i][j].split("/").pop().split(".")[0];
      object.traverse((child) => {
        if (child instanceof THREE.Mesh) {
          child.material = new THREE.MeshBasicMaterial({
            side: THREE.DoubleSide,
            color: 0x000000,
          });
        }
      });
      scene.add(object);
      if(!meshes){
        meshes = object;
      }else{
        meshes.add(object);
      }

    }
    // Get Bounding Box Center of the meshes
    let boundingBox = new THREE.Box3().setFromObject(meshes);
    let boundingBoxCenter = new THREE.Vector3();
    boundingBox.getCenter(boundingBoxCenter);
    console.log('boundingBoxCenter:', boundingBoxCenter);

    let T0_Ti = new THREE.Matrix4(); //# T0_Ti = T0_Cam @ Cam_Ti
    let Cam_Ti = new THREE.Matrix4()
    .fromArray(extrinsics[i].flat())
    .transpose()
    console.log('Ti extrinsics:', extrinsics[i].flat(), i );

    let Cam_T0 = new THREE.Matrix4()
    .fromArray(extrinsics[0].flat())
    .transpose()
    T0_Ti = Cam_T0.clone().invert().multiply(Cam_Ti); // T0_Ti =T0_Cam @ Cam_Ti

    let T0_Ti_translation = new THREE.Vector3();
    let T0_Ti_rotation = new THREE.Quaternion();
    T0_Ti.decompose(T0_Ti_translation, T0_Ti_rotation, new THREE.Vector3());
    console.log('T0_Ti:', T0_Ti, i);


    
    currentBBoxToCam[objectName] = T0_Ti.clone().setPosition(new THREE.Vector3(boundingBoxCenter.x, boundingBoxCenter.y, boundingBoxCenter.z));  //bbox_cam
    currentMeshToBBox[objectName] = currentBBoxToCam[objectName].clone().invert(); // bbox_cam = cam_bbox @ T0_Ti^-1
    console.log('currentBBoxToCam:', currentBBoxToCam[objectName]);
    
    if (displayAllCoordinates || i === selectedObjectIndex) {
      // Get center of mesh
      const axesHelper = new THREE.AxesHelper(15);
      axesHelper.matrix = currentBBoxToCam[objectName].clone();  // cam_bbox
      axesHelper.matrixAutoUpdate = false;  // Disable auto updating of the matrix

      // Update the matrixWorld in case it is needed
      axesHelper.updateMatrixWorld(true);

      //  Add the AxesHelper to the scene
      scene.add(axesHelper);

    }
    
    
    
    const objectDropdown = document.getElementById("objectDropdown");
    objectDropdown.innerHTML = ""; // Clear existing options
    for (let i = 0; i < objPaths.length; i++) {
      // Create a new option element
      const option = document.createElement("option");
      option.value = i;
      let text = '';
      for (let j = 0; j < objPaths[i].length; j++) {
        text = text + 'Obj_' + objPaths[i][j].split("/").pop().split(".")[0] + ' ';
      }
      option.text = text;
      objectDropdown.appendChild(option); // Add the option to the dropdown
    }
    

  }
  if (displayAllCoordinates) {
    const axesHelper = new THREE.AxesHelper(15);
    scene.add(axesHelper);
  }

}

//////////////////////////// Transformation ////////////////////////////
const quickRotateXButtons = document.querySelectorAll('.quick-rotate-x');
quickRotateXButtons.forEach(button => {
  button.addEventListener('click', () => {
    const angle = parseFloat(button.dataset.value);
    // rotateXSlider.value = angle;
    rotateXValue.value = angle;
    transformObject(objPaths[selectedObjectIndex], angle, [1, 0, 0], [0, 0, 0]);
  });
});

const quickRotateYButtons = document.querySelectorAll('.quick-rotate-y');
quickRotateYButtons.forEach(button => {
  button.addEventListener('click', () => {
    const angle = parseFloat(button.dataset.value);
    // rotateYSlider.value = angle;
    rotateYValue.value = angle;
    transformObject(objPaths[selectedObjectIndex], angle, [0, 1, 0], [0, 0, 0]);
  });
});

const quickRotateZButtons = document.querySelectorAll('.quick-rotate-z');
quickRotateZButtons.forEach(button => {
  button.addEventListener('click', () => {
    const angle = parseFloat(button.dataset.value);
    // rotateZSlider.value = angle;
    rotateZValue.value = angle;
    transformObject(objPaths[selectedObjectIndex], angle, [0, 0, 1], [0, 0, 0]);
  });
});

const quickTranslateXButtons = document.querySelectorAll('.quick-translate-x');
quickTranslateXButtons.forEach(button => {
  button.addEventListener('click', () => {
    const distance = parseFloat(button.dataset.value);
    // translateXSlider.value = distance;
    translateXValue.value = distance;
    transformObject(objPaths[selectedObjectIndex], 0, [0, 0, 0], [distance, 0, 0]);
  });
});

const quickTranslateYButtons = document.querySelectorAll('.quick-translate-y');  
quickTranslateYButtons.forEach(button => {
  button.addEventListener('click', () => {
    const distance = parseFloat(button.dataset.value);
    // translateYSlider.value = distance;
    translateYValue.value = distance;
    transformObject(objPaths[selectedObjectIndex], 0, [0, 0, 0], [0, distance, 0]);
  });
});

const quickTranslateZButtons = document.querySelectorAll('.quick-translate-z');
quickTranslateZButtons.forEach(button => {
  button.addEventListener('click', () => {
    const distance = parseFloat(button.dataset.value);
    // translateZSlider.value = distance;
    translateZValue.value = distance;
    transformObject(objPaths[selectedObjectIndex], 0, [0, 0, 0], [0, 0, distance]);
  });
});

////////////////////// Transform Object ////////////////////////////
const rotateXValue = document.getElementById("rotateXValue");
const rotateYValue = document.getElementById("rotateYValue");
const rotateZValue = document.getElementById("rotateZValue");

rotateXValue.addEventListener("input", () => {
  const angle = parseFloat(rotateXValue.value);
  // rotateXSlider.value = angle;
  transformObject(objPaths[selectedObjectIndex], angle, [1, 0, 0], [0, 0, 0]);
});

rotateYValue.addEventListener("input", () => {
  const angle = parseFloat(rotateYValue.value);
  // rotateYSlider.value = angle;
  transformObject(objPaths[selectedObjectIndex], angle, [0, 1, 0], [0, 0, 0]);
});

rotateZValue.addEventListener("input", () => {
  const angle = parseFloat(rotateZValue.value);
  // rotateZSlider.value = angle;
  transformObject(objPaths[selectedObjectIndex], angle, [0, 0, 1], [0, 0, 0]);
});

const translateXValue = document.getElementById("translateXValue");
const translateYValue = document.getElementById("translateYValue");
const translateZValue = document.getElementById("translateZValue");

translateXValue.addEventListener("input", () => {
  const distance = parseFloat(translateXValue.value);
  transformObject(objPaths[selectedObjectIndex], 0, [0, 0, 0], [distance, 0, 0]);
});

translateYValue.addEventListener("input", () => {
  const distance = parseFloat(translateYValue.value);
  transformObject(objPaths[selectedObjectIndex], 0, [0, 0, 0], [0, distance, 0]);
});

translateZValue.addEventListener("input", () => {
  const distance = parseFloat(translateZValue.value);
  transformObject(objPaths[selectedObjectIndex], 0, [0, 0, 0], [0, 0, distance]);
});

////////////////////////////////////////// Transform Object ////////////////////////////
async function transformObject(objPath, rotationAngle, rotationAxis, translation) {
  if (transformationLock) {
    console.log("Transformation in progress, please wait...");
    await new Promise(resolve => setTimeout(resolve, 100));
  }

  console.log("Transforming object:", objPath);
  console.log("Rotation angle:", rotationAngle);
  console.log("Rotation axis:", rotationAxis);
  console.log("Translation:", translation);

  try {
    set_transformation_lock(true);
    const objectName = 'obj_' + objPath[0].split("/").pop().split(".")[0];

    if(!currentTransfomations[objectName]){
      currentTransfomations[objectName] = new THREE.Matrix4();
    }

    // Calculate the bounding box center of the object
    let appliedTransfomation = new THREE.Matrix4();
  
    let deltaAngle = [rotationAngle*rotationAxis[0], rotationAngle *rotationAxis[1], rotationAngle*rotationAxis[2]];
    let deltaTranslation = [translation[0] * translationScale, translation[1] * translationScale, translation[2] * translationScale];

    console.log('deltaAngle:', deltaAngle);
    console.log('deltaTranslation:', deltaTranslation);

    appliedTransfomation.makeRotationFromEuler(new THREE.Euler(deltaAngle[0] * Math.PI / 180, deltaAngle[1] * Math.PI / 180, deltaAngle[2] * Math.PI / 180));
    appliedTransfomation.setPosition(new THREE.Vector3(deltaTranslation[0], deltaTranslation[1], deltaTranslation[2])); //bbox_bbox'
    
    // Update the current object's translation
    
    newBBoxToBBox = appliedTransfomation.clone();  // bbox_bbox'
    let MeshToBBox = currentMeshToBBox[objectName].clone();  // bbox_mesh
    let newBBoxToCam = currentBBoxToCam[objectName].clone();  // cam_bbox
    newBBoxToCam.multiply(newBBoxToBBox); //cam_bbox' = cam_bbox @ bbox_bbox'
    // Set rotation to 0
    console.log('set BBoxToCam from:', currentBBoxToCam[objectName] , 'to:', newBBoxToCam);
    currentBBoxToCam[objectName] = newBBoxToCam.clone();  // cam_bbox <-- cam_bbox'


    let MeshToCam = newBBoxToCam.clone();  // cam_bbox', cam_mesh init
    // cam_mesh = cam_bbox @ bbox_mesh
    MeshToCam.multiply(MeshToBBox);

    console.log('appliedTransfomation:', appliedTransfomation);
    
    currentTransfomations[objectName] = MeshToCam;

    let currentRotation = new THREE.Euler();
    let currentTranslation = new THREE.Vector3();
    currentTransfomations[objectName].decompose(currentTranslation, currentRotation, new THREE.Vector3());
    console.log('decomposate currentRotation:', currentRotation);
    console.log('decomposate currentTranslation:', currentTranslation);

    await reloadScene();
    await set_transformation_lock(false);

  } catch (error) {
    console.error("Error transforming object:", error);
  }
}

///////////////////////////////////////// Transform All Objects ////////////////////////////
// const rotateXAllSlider = document.getElementById("rotateXAllSlider");
const rotateXAllValue = document.getElementById("rotateXAllValue");
const rotateYAllValue = document.getElementById("rotateYAllValue");
const rotateZAllValue = document.getElementById("rotateZAllValue");
const translateXAllValue = document.getElementById("translateXAllValue");
const translateYAllValue = document.getElementById("translateYAllValue");
const translateZAllValue = document.getElementById("translateZAllValue");


rotateXAllValue.addEventListener("input", () => {
  const angle = parseFloat(rotateXAllValue.value);
  transformAllObject( angle, [1, 0, 0], [0, 0, 0]);
});

rotateYAllValue.addEventListener("input", () => {
  const angle = parseFloat(rotateYAllValue.value);
  transformAllObject( angle, [0, 1, 0], [0, 0, 0]);
});

rotateZAllValue.addEventListener("input", () => {
  const angle = parseFloat(rotateZAllValue.value);
  transformAllObject(angle, [0, 0, 1], [0, 0, 0]);
});

translateXAllValue.addEventListener("input", () => {
  const distance = parseFloat(translateXAllValue.value);
  transformAllObject( 0, [0, 0, 0], [distance, 0, 0]);
});

translateYAllValue.addEventListener("input", () => {
  const distance = parseFloat(translateYAllValue.value);
  transformAllObject( 0, [0, 0, 0], [0, distance, 0]);
});

translateZAllValue.addEventListener("input", () => {
  const distance= parseFloat(translateZAllValue.value);
  transformAllObject( 0, [0, 0, 0], [0, 0, distance]);
});

const quickRotateAllXButtons = document.querySelectorAll('.quick-rotate-all-x');
quickRotateAllXButtons.forEach(button => {
  button.addEventListener('click', () => {
    const angle = parseFloat(button.dataset.value);
    rotateXAllValue.value = angle;
    transformAllObject( angle, [1, 0, 0], [0, 0, 0]);
  });
});

const quickRotateAllYButtons = document.querySelectorAll('.quick-rotate-all-y');
quickRotateAllYButtons.forEach(button => {
  button.addEventListener('click', () => {
    const angle = parseFloat(button.dataset.value);
    rotateYAllValue.value = angle;
    transformAllObject( angle, [0, 1, 0], [0, 0, 0]);
  });
});

const quickRotateAllZButtons = document.querySelectorAll('.quick-rotate-all-z');
quickRotateAllZButtons.forEach(button => {
  button.addEventListener('click', () => {
    const angle = parseFloat(button.dataset.value);
    rotateZValue.value = angle;
    transformAllObject(angle, [0, 0, 1], [0, 0, 0]);
  });
});

const quickTranslateAllXButtons = document.querySelectorAll('.quick-translate-all-x');
quickTranslateAllXButtons.forEach(button => {
  button.addEventListener('click', () => {
    const distance = parseFloat(button.dataset.value);
    translateXAllValue.value = distance;
    transformAllObject(0, [0, 0, 0], [distance, 0, 0]);
  });
});

const quickTranslateAllYButtons = document.querySelectorAll('.quick-translate-all-y');
quickTranslateAllYButtons.forEach(button => {
  button.addEventListener('click', () => {
    const distance = parseFloat(button.dataset.value);
    translateYAllValue.value = distance;
    transformAllObject(0, [0, 0, 0], [0, distance, 0]);
  });
});

const quickTranslateAllZButtons = document.querySelectorAll('.quick-translate-all-z');
quickTranslateAllZButtons.forEach(button => {
  button.addEventListener('click', () => {
    const distance = parseFloat(button.dataset.value);
    translateZAllValue.value = distance;
    transformAllObject(0, [0, 0, 0], [0, 0, distance]);
  });
});


function set_transformation_lock(value){
  transformationLock = value;
  if(value){
    rotateXAllValue.disabled = true;
    rotateYAllValue.disabled = true;
    rotateZAllValue.disabled = true;
    translateXAllValue.disabled = true;
    translateYAllValue.disabled = true;
    translateZAllValue.disabled = true;
    rotateXValue.disabled = true;
    rotateYValue.disabled = true;
    rotateZValue.disabled = true;
    translateXValue.disabled = true;
    translateYValue.disabled = true;
    translateZValue.disabled = true;
  }else{
    rotateXAllValue.disabled = false;
    rotateYAllValue.disabled = false;
    rotateZAllValue.disabled = false;
    translateXAllValue.disabled = false;
    translateYAllValue.disabled = false;
    translateZAllValue.disabled = false;
    rotateXValue.disabled = false;
    rotateYValue.disabled = false;
    rotateZValue.disabled = false;
    translateXValue.disabled = false;
    translateYValue.disabled = false;
    translateZValue.disabled = false;
  
  }

}


////////////////////////// Transform All Objects ////////////////////////////
async function transformAllObject(rotationAngle, rotationAxis, translation) {
  console.log("Transforming all objects...");
  if (transformationLock) {
    console.log("Transformation in progress, please wait...");
    await new Promise(resolve => setTimeout(resolve, 100));
  }

  rotationAngle = -rotationAngle;
  translation = [-translation[0], -translation[1], -translation[2]];

  set_transformation_lock(true);
  for(let j=0; j<objPaths.length; j++){
    console.log('Transforming object:', objPaths[j], 'for total:', objPaths.length);
    const objPath = objPaths[j];
    try {
      const objectName = 'obj_' + objPath[0].split("/").pop().split(".")[0];

      if(!currentTransfomations[objectName]){
        currentTransfomations[objectName] = new THREE.Matrix4();
      }

      // Calculate the bounding box center of the object
      let appliedTransfomation = new THREE.Matrix4();

      let deltaAngle = [rotationAngle*rotationAxis[0], rotationAngle *rotationAxis[1], rotationAngle*rotationAxis[2]];
      let deltaTranslation = [translation[0] * translationScale, translation[1] * translationScale, translation[2] * translationScale];

      console.log('deltaAngle:', deltaAngle);
      console.log('deltaTranslation:', deltaTranslation);

      appliedTransfomation.makeRotationFromEuler(new THREE.Euler(deltaAngle[0] * Math.PI / 180, deltaAngle[1] * Math.PI / 180, deltaAngle[2] * Math.PI / 180));
      appliedTransfomation.setPosition(new THREE.Vector3(deltaTranslation[0], deltaTranslation[1], deltaTranslation[2])); //bbox_bbox'
      
      let newMeshToCam = appliedTransfomation.clone().invert();  // cam'_cam
      newMeshToCam.multiply(currentTransfomations[objectName]); // cam'_mesh = cam'_cam @ cam_mesh
      currentTransfomations[objectName] = newMeshToCam.clone();  // cam_mesh <-- cam'_mesh

      // currentBBoxToCam[objectName] = currentTransfomations[objectName].clone();
      let newCurrentBBoxToCam = appliedTransfomation.clone().invert(); // appliedTransfomation: cam_cam', currentBBoxToCam: cam_bbox
      newCurrentBBoxToCam.multiply(currentBBoxToCam[objectName]); // cam'_bbox = cam'_cam @ cam_bbox
      currentBBoxToCam[objectName] = newCurrentBBoxToCam.clone();  // bbox_cam <-- bbox_cam'

      let currentRotation = new THREE.Euler();
      let currentTranslation = new THREE.Vector3();
      currentTransfomations[objectName].decompose(currentTranslation, currentRotation, new THREE.Vector3());
      console.log('decomposate currentRotation:', currentRotation);
      console.log('decomposate currentTranslation:', currentTranslation);
      
      
    } catch (error) {
      console.error("Error transforming object:", error);
    }
  }
  await reloadScene();
  await set_transformation_lock(false);
}

async function reloadObjects() {
  console.log("Reloading objects...", objPaths.length);
  
  for (let i = 0; i < objPaths.length; i++) {
    const objectName = 'obj_' + objPaths[i][0].split("/").pop().split(".")[0];
    let meshes;
    for (let j = 0; j < objPaths[i].length; j++) {
      
      const loader = new OBJLoader();
      const object = await loader.loadAsync(objPaths[i][j]);
      object.name = objectName;


      // Transform the object based on current transformations
      if(currentTransfomations[objectName]){
        object.applyMatrix4(currentTransfomations[objectName]);
      }

      // Set the object's color based on the selection
      object.traverse((child) => {
        if (child instanceof THREE.Mesh) {
          child.material = new THREE.MeshBasicMaterial({
            color: i === selectedObjectIndex ? 0xA00000 : 0x000000,
            side: THREE.DoubleSide,
          });
        }
      });

      scene.add(object);
      if(!meshes){
        meshes = object;
      }else{
        meshes.add(object);
      }
    }
    // Get Bounding Box Center of the meshes
    let boundingBox = new THREE.Box3().setFromObject(meshes);
    let boundingBoxCenter = new THREE.Vector3();
    boundingBox.getCenter(boundingBoxCenter);
    console.log('updated boundingBoxCenter:', boundingBoxCenter);

    if (displayAllCoordinates || i === selectedObjectIndex) {
     // Apply the matrix
     let axesHelper = new THREE.AxesHelper(15);
     axesHelper.matrix = currentBBoxToCam[objectName];
     axesHelper.matrixAutoUpdate = false;  // Disable auto updating of the matrix

     // Update the matrixWorld in case it is needed
     axesHelper.updateMatrixWorld(true);
     // Add the AxesHelper to the scene
     scene.add(axesHelper);
    }
  }

  if (displayAllCoordinates) {
    const axesHelper = new THREE.AxesHelper(15);
    scene.add(axesHelper);
  }


}
/////////// Save Transformation ////////////
async function saveTransformation() {
  currentRotations = {};
  currentTranslations = {};
  for(let i = 0; i < objPaths.length; i++){
    const objectName = 'obj_' + objPaths[i][0].split("/").pop().split(".")[0];
    let currentRotation = new THREE.Euler();
    let currentTranslation = new THREE.Vector3();
    currentTransfomations[objectName].decompose(currentTranslation, currentRotation, new THREE.Vector3());

    currentRotations[objectName] = [currentRotation.x * 180 / Math.PI, currentRotation.y * 180 / Math.PI, currentRotation.z * 180 / Math.PI];
    currentTranslations[objectName] = [currentTranslation.x, currentTranslation.y, currentTranslation.z];
    console.log('currentRotation:', currentRotation, 'currentTranslation:', currentTranslation);
  }

  console.log("Current Rotations:", currentRotations);
  console.log("Current Translations:", currentTranslations);

  try {
      const response = await fetch(`${host}/transform-objects?category=${selectedCategory}&name=${selectedName}&video_id=${selectedVideoId}&frame_id=${selectedFrameId}&user_id=${selectedUserId}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          objPaths: objPaths,
          rotations: currentRotations,
          translations: currentTranslations,
        }),
      });
      console.log(await response.text());
    
  } catch (error) {
    console.error("Error saving transformation:", error);
  }
  //// After all thing finished, reload the objects
  await reloadObjectsAndCam();

  console.log("Resetting the pose...");

  const rotateXValue = document.getElementById("rotateXValue");
  const rotateYValue = document.getElementById("rotateYValue");
  const rotateZValue = document.getElementById("rotateZValue");

  const translateXValue = document.getElementById("translateXValue");
  const translateYValue = document.getElementById("translateYValue");
  const translateZValue = document.getElementById("translateZValue");


  rotateXValue.value = 0;
  rotateYValue.value = 0;
  rotateZValue.value = 0;

  translateXValue.value = 0;
  translateYValue.value = 0;
  translateZValue.value = 0;

  
//   const promises = [
//     // fetchObjPaths(selectedCategory, selectedName, selectedVideoId, selectedFrameId),
//     // loadObjects(),
//     loadOverlayImage(selectedCategory, selectedName, selectedVideoId, selectedFrameId, selectedUserId),
//     displayMasks(selectedCategory, selectedName, selectedVideoId, selectedFrameId)
// ];
//   await Promise.all(promises);
//   console.log("Overlay Image Loaded");
  console.log('Pose Reset');

  getProgressBar();
  loadOverlayImage(selectedCategory, selectedName, selectedVideoId, selectedFrameId, selectedUserId)

}
const savePoseButton = document.getElementById("savePoseButton");
savePoseButton.addEventListener("click", saveTransformation);


async function reloadObjectsAndCam() {

   // Wait for the selected values to be set
   while (!selectedCategory || !selectedName || !selectedVideoId || !selectedFrameId) {
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  console.log('selectedCategory:', selectedCategory, 'selectedName:', selectedName, 'selectedVideoId:', selectedVideoId, 'selectedFrameId:', selectedFrameId);
  await fetchCameraParametersAndObjPaths(selectedCategory, selectedName, selectedVideoId, selectedFrameId);
  console.log("Camera Parameters Loaded");
  sceneCanvas = document.getElementById("sceneCanvas");
  imageCanvas = document.getElementById("imageCanvas");
  imageContext = imageCanvas.getContext("2d");

  sceneCanvas.width = img_width  * renderResizeRate;
  sceneCanvas.height = img_height   * renderResizeRate;
  imageCanvas.width = sceneCanvas.width;
  imageCanvas.height = sceneCanvas.height;
  
  await setCameraView();
  // Scene
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x91cf93);

  await loadObjects();
  
}
////////////////Get MSE Score//////////////////
// const getMSEButton = document.getElementById("getMSEScoresButton");
// getMSEButton.addEventListener("click", getMSE);

// async function getMSE() {
//   console.log("Getting MSE scores...");
//   await saveTransformation();

//   try {
//     const response = await fetch(`${host}/get-mse-score?category=${selectedCategory}&name=${selectedName}&video_id=${selectedVideoId}&frame_id=${selectedFrameId}&user_id=${selectedUserId}`);
//     const data = await response.json();
//     mseScores = data.mseScores;
//     console.log('data:', data);
//   }
//   catch(error){
//     console.error("Error getting MSE scores:", error);
//   }
//   const mseScore = document.getElementById("mseScore");
//   for (let i = 0; i < objPaths.length; i++) {
//     mseScore.innerHTML += `Obj_${i}: ${mseScores[i].toFixed(2)}<br>`;
//   }
// }

////////////////////Get progress /////////////////
async function getProgressBar(){
  try{
    console.log('Getting progress')
    // Get progress bar from backend
    const response = await fetch(`${host}/progress-bar?category=${selectedCategory}&name=${selectedName}&video_id=${selectedVideoId}&frame_id=${selectedFrameId}&user_id=${selectedUserId}`);
    const data = await response.json();
    console.log('data:', data);
    const progress = data.progress;
    const annotated = data.annotated;
    const progressImage = data.progressImage;
    const annotatedImage = data.annotatedImage;
    const totalImage  = data.totalImage;
    const total  = data.total;
    const message = data.message;
    console.log('progress:', progress, 'annotated:', annotated, 'total:', total)
    progressBar.style.width = `${progress}%`;
    console.log('progress:', progress, 'annotated:', annotated, 'total:', total)
    progressNumber.textContent = `${annotated}/${total}`
    progressBarImage.style.width = `${progressImage}%`;
    progressNumberImage.textContent = `${annotatedImage}/${totalImage}`
    progressTextarea.textContent = data.message;
    console.log('Message:', message);
    

  } catch (error) {
    console.error("Error fetching progress");
    // return null;
  }

}

////////////////Load from prev pose ////////////////////
const usePoseFromPrevFrameButton = document.getElementById("usePoseFromPrevFrameButton");
usePoseFromPrevFrameButton.addEventListener("click", usePoseFromPrevFrame);

async function usePoseFromPrevFrame() {
  console.log("Using pose from previous frame...");

  try{
    const response = await fetch(`${host}/load-pose-from-the-prev-frame?category=${selectedCategory}&name=${selectedName}&video_id=${selectedVideoId}&frame_id=${selectedFrameId}&user_id=${selectedUserId}`);
    const data = await response.json();
    console.log('data:', data);
  }
  catch(error){
    console.error("Error loading pose from the previous frame:", error);
  }
  await reloadObjectsAndCam();

  console.log('Pose loaded from the previous frame');
}
///////////////////////// Progress bar ////////////
const progressBar = document.querySelector('.progress-bar');
const progressNumber = document.querySelector('.progress-number');
const progressBarImage = document.querySelector('.progress-bar-image');
const progressNumberImage = document.querySelector('.progress-number-image');
const progressTextarea = document.getElementById('progressTextarea');
///////////////// Reset Pose ////////////////////
async function resetPose() {
  console.log("Resetting the pose...");

  const rotateXValue = document.getElementById("rotateXValue");
  const rotateYValue = document.getElementById("rotateYValue");
  const rotateZValue = document.getElementById("rotateZValue");

  const translateXValue = document.getElementById("translateXValue");
  const translateYValue = document.getElementById("translateYValue");
  const translateZValue = document.getElementById("translateZValue");


  rotateXValue.value = 0;
  rotateYValue.value = 0;
  rotateZValue.value = 0;

  translateXValue.value = 0;
  translateYValue.value = 0;
  translateZValue.value = 0;


  for( let i = 0; i < objPaths.length; i++){
    const objectName = 'obj_' + objPaths[i][0].split("/").pop().split(".")[0];
    currentTransfomations[objectName] = new THREE.Matrix4();
    currentBBoxToCam[objectName] = new THREE.Matrix4();
    
  }

  // await reloadScene();
  while (scene.children.length > 0) {
    scene.remove(scene.children[0]);
  }
  await loadObjects();
  console.log('Pose Reset');

  getProgressBar();

}
const resetPoseButton = document.getElementById("resetPoseButton");
resetPoseButton.addEventListener("click", resetPose);




//////////////////////////// Object Selection ////////////////////////////
let selectedObjectIndex = 0;
const objectDropdown = document.getElementById("objectDropdown");
objectDropdown.addEventListener("change", function () {
  console.log("Selected object:", this.value);

  // if this.value not defined
  if (this.value === undefined) {
    selectedObjectIndex = 0;
  } else {

    selectedObjectIndex = parseInt(this.value);
  }
  reloadScene();
  console.log('selectedObjectIndex' , selectedObjectIndex)

  console.log('Resetting Mask Container')
  const maskContainer = document.getElementById("maskContainer");
  

  for (let i = 0; i < maskContainer.children.length; i++) {
    const img = maskContainer.children[i].children[0];
    if (i === selectedObjectIndex) {
      img.classList.add("selected");
    } else {
      img.classList.remove("selected");
    }
  }
});

//////////////////////////// Reload Scene ////////////////////////////
async function reloadScene() {
  // Clear the existing scene
  while (scene.children.length > 0) {
    
    scene.remove(scene.children[0]);
  }
  // Reload the objects
  await reloadObjects();
  
}

////////////////// Reset to Camera View//////////////////
const resetCameraButton = document.getElementById("resetCameraButton");
resetCameraButton.addEventListener("click", resetCamera);

function resetCamera() {
  console.log("Resetting the camera view...");

  // Reset the camera view
  setCameraView();

  
}
//////////////////////////// Image Overlay ////////////////////////////

async function fetchOverlayImage(selectedCategory, selectedName, selectedVideoId, selectedFrameId, selectedUserId) {
  try {
    const response = await fetch(`${host}/overlay-image?category=${selectedCategory}&name=${selectedName}&video_id=${selectedVideoId}&frame_id=${selectedFrameId}&user_id=${selectedUserId}`);
    const data = await response.json();
    return data.imgPath;
  } catch (error) {
    console.error("Error fetching overlay image:", error);
    return null;
  }
}

async function loadOverlayImage(selectedCategory, selectedName, selectedVideoId, selectedFrameId, selectedUserId) {
  const imageData = await fetchOverlayImage(selectedCategory, selectedName, selectedVideoId, selectedFrameId, selectedUserId);
  if (!imageData) return;

  const image = new Image();
  image.src = imageData;
  image.onload = function() {
    console.log("Overlay image loaded");
    console.log("Image dimensions:", image.width, "x", image.height);
    imageContext.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
    imageContext.drawImage(image, 0, 0, imageCanvas.width, imageCanvas.height);
    updateImageOpacity(); // Set the initial opacity
  };
}

function updateImageOpacity() {
  imageCanvas.style.opacity = opacitySlider.value;
}


//////////////////////////// Camera View ////////////////////////////
async function setCameraView() {

  // Init camera
  let fov = 2 * Math.atan(img_height / (2 * intrinsics[1][1])) * (180 / Math.PI);
  camera = new THREE.PerspectiveCamera(fov, img_width / img_height, 0.1, 1000);

  // Init OrbitControls
  controls = new OrbitControls(camera, renderer.domElement);
  controls.rotateSpeed = 0.3;
  console.log('Control Setted')

  // Update the camera matrix
  modelMatrix = new THREE.Matrix4()
    .fromArray(extrinsics[0].flat())
    .transpose()
    .invert();
  modelMatrix.elements.forEach((value, index) => {
    if (index > 3 && index < 12) modelMatrix.elements[index] = -value;
  });

  let position = new THREE.Vector3().setFromMatrixPosition(modelMatrix);
  let scale = new THREE.Vector3().setFromMatrixScale(modelMatrix);
  let upVector = new THREE.Vector3().setFromMatrixColumn(modelMatrix, 1);
  let distance = new THREE.Vector3()
    .subVectors(position, controls.target)
    .length();
  let direction = new THREE.Vector3(0, 0, -1).transformDirection(modelMatrix);

  controls.target = new THREE.Vector3()
    .add(direction.multiplyScalar(distance))
    .add(position);
  camera.lookAt(controls.target);

  camera.matrixAutoUpdate = false;
  camera.up.copy(upVector);
  camera.position.copy(position);
  camera.scale.copy(scale);
  camera.updateMatrix();
  camera.matrixAutoUpdate = true;

  controls.update();
  camera.updateMatrix();

  controls.addEventListener("change", () => {
    renderer.render(scene, camera);
  });

  
}


/////////////////// Load Masks ///////////////////////////
async function fetchMaskImages(selectedCategory, selectedName, selectedVideoId, selectedFrameId) {
  try {
    const response = await fetch(`${host}/masks-data?category=${selectedCategory}&name=${selectedName}&video_id=${selectedVideoId}&frame_id=${selectedFrameId}&user_id=${selectedUserId}`);
    const data = await response.json();
    return data.masksPaths
  } catch (error) {
    console.error("Error fetching mask paths:", error);
    return [];
  }
}

async function displayMasks(category, name, video_id, frame_id, userId) {
  const maskImages = await fetchMaskImages(category, name, video_id, frame_id);
  const maskContainer = document.getElementById("maskContainer");
  maskContainer.innerHTML = "";

  const aspectRatio = img_width / img_height;
  const maxWidth = 300; // Adjust this value as needed
  const maxHeight = maxWidth / aspectRatio;

  maskImages.forEach((maskImage, index) => {
    const imgContainer = document.createElement("div");
    imgContainer.style.display = "inline-block";
    imgContainer.style.width = `${maxWidth}px`;
    imgContainer.style.height = `${maxHeight}px`;
    imgContainer.style.margin = "5px";

    const img = document.createElement("img");
    img.src = maskImage;
    img.classList.add("mask-image");
    if (index === selectedObjectIndex) {
      img.classList.add("selected");
    }
    img.style.width = "100%";
    img.style.height = "100%";
    img.style.objectFit = "contain";

    imgContainer.appendChild(img);
    maskContainer.appendChild(imgContainer);
  });
}


//////////////////// Load Cat, Name, Video_id, Frame_id ////////////////////
async function fetchUserIds() {
  try {
    const response = await fetch(`${host}/user_ids`);
    const data = await response.json();
    return data.userIds;
  } catch (error) {
    console.error("Error fetching user IDs:", error);
    return [];
  }
}
async function fetchCategories(userId) {
  try {
    const response = await fetch(`${host}/categories?user_id=${userId}`);
    const data = await response.json();
    return data.categories;
  } catch (error) {
    console.error("Error fetching categories:", error);
    return [];
  }
}

async function fetchNames(userId, category) {
  try {
    const response = await fetch(`${host}/names?category=${category}&user_id=${userId}`);
    const data = await response.json();
    return data.names;
  } catch (error) {
    console.error("Error fetching names:", error);
    return [];
  }
}

async function fetchFrameIds(userId, category, name, videoId) {
  try {
    const response = await fetch(`${host}/frame-ids?category=${category}&name=${name}&video_id=${videoId}&user_id=${userId}`);
    const data = await response.json();
    return data.frameIds;
  } catch (error) {
    console.error("Error fetching frame IDs:", error);
    return [];
  }
}

async function fetchVideoIds(userId, category, name) {
  try {
    const response = await fetch(`${host}/video-ids?category=${category}&name=${name}&user_id=${userId}`);
    const data = await response.json();
    return data.videoIds;
  } catch (error) {
    console.error("Error fetching video IDs:", error);
    return [];
  }
}

async function populateDropdowns() {
  const userIdDropdown = document.getElementById("userIdDropdown");
  const categoryDropdown = document.getElementById("categoryDropdown");
  const nameDropdown = document.getElementById("nameDropdown");
  const frameIdDropdown = document.getElementById("frameIdDropdown");
  const videoIdDropdown = document.getElementById("videoIdDropdown");

  const userIds = await fetchUserIds();
  userIds.forEach((userid) => {
    const option = document.createElement("option");
    option.value = userid;
    option.text = userid;
    userIdDropdown.appendChild(option);
  });
  console.log("User IDs loaded", userIds);

  // Set the selected user ID to the first user ID in the list
  selectedUserId = userIds[0];
  userId = selectedUserId;
  
  const categories = await fetchCategories(selectedUserId);
  categories.forEach((name) => {
    const option = document.createElement("option");
    option.value = name;
    option.text = name;
    categoryDropdown.appendChild(option);
  });

  // Set the selected category to the first category in the list
  selectedCategory = categories[0];
  category = selectedCategory;

  const names = await fetchNames(selectedUserId, selectedCategory);
  names.forEach((name) => {
    const option = document.createElement("option");
    option.value = name;
    option.text = name;
    nameDropdown.appendChild(option);
  });

  // Set the selected name to the first name in the list
  selectedName = names[0];
  name = selectedName;

  const videoIds = await fetchVideoIds(selectedUserId, selectedCategory, selectedName);
  videoIds.forEach((videoId) => {
    const option = document.createElement("option");
    option.value = videoId;
    option.text = videoId;
    videoIdDropdown.appendChild(option);
  });

  // Set the selected video ID to the first video ID in the list
  selectedVideoId = videoIds[0];
  video_id = selectedVideoId;

  const frameIds = await fetchFrameIds(selectedUserId, selectedCategory, selectedName, selectedVideoId);
  frameIds.forEach((frameId) => {
    const option = document.createElement("option");
    option.value = frameId;
    option.text = frameId;
    frameIdDropdown.appendChild(option);
  });

  // Set the selected frame ID to the first frame ID in the list
  selectedFrameId = frameIds[0];
  frame_id = selectedFrameId;

  userIdDropdown.addEventListener("change", async function () {
    console.log("Selected user ID:", this.value);
    selectedUserId = this.value;
    const categories = await fetchCategories(selectedUserId);
    categoryDropdown.innerHTML = "";
    categories.forEach((name) => {
      const option = document.createElement("option");
      option.value = name;
      option.text = name;
      categoryDropdown.appendChild(option);
    });
    categoryDropdown.dispatchEvent(new Event("change"));
    userId = selectedUserId;
  });
  

  categoryDropdown.addEventListener("change", async function () {
    console.log("Selected category:", this.value);
    selectedCategory = this.value;
    const names = await fetchNames(selectedUserId, selectedCategory);
    nameDropdown.innerHTML = "";
    names.forEach((name) => {
      const option = document.createElement("option");
      option.value = name;
      option.text = name;
      nameDropdown.appendChild(option);
    });
    nameDropdown.dispatchEvent(new Event("change"));
    category = selectedCategory;
  });

  nameDropdown.addEventListener("change", async function () {
    console.log("Selected name:", this.value);
    selectedCategory = categoryDropdown.value;
    selectedName = this.value;
    const videoIds = await fetchVideoIds(selectedUserId,selectedCategory, selectedName);
    videoIdDropdown.innerHTML = "";
    videoIds.forEach((videoId) => {
      const option = document.createElement("option");
      option.value = videoId;
      option.text = videoId;
      videoIdDropdown.appendChild(option);
    });
    videoIdDropdown.dispatchEvent(new Event("change"));
    name = selectedName;
  });

  videoIdDropdown.addEventListener("change", async function () {
    console.log("Selected video ID:", this.value);
    selectedCategory = categoryDropdown.value;
    selectedName = nameDropdown.value;
    selectedVideoId = this.value;
    const frameIds = await fetchFrameIds(selectedUserId,selectedCategory, selectedName, selectedVideoId);
    frameIdDropdown.innerHTML = "";
    frameIds.forEach((frameId) => {
      const option = document.createElement("option");
      option.value = frameId;
      option.text = frameId;
      frameIdDropdown.appendChild(option);
    });
    frameIdDropdown.dispatchEvent(new Event("change"));
    video_id = selectedVideoId;
  });

  frameIdDropdown.addEventListener("change", async function () {
    console.log("Selected frame ID:", this.value);
    selectedCategory = categoryDropdown.value;
    selectedName = nameDropdown.value;
    selectedFrameId = this.value;
    selectedVideoId = videoIdDropdown.value;

    frame_id = selectedFrameId;

    await init();
  });

  // Trigger change event on the category dropdown to populate other dropdowns
  categoryDropdown.dispatchEvent(new Event("change"));
}

//////////////////// Initialize the Scene ////////////////////
async function init() {
  //////////////////////////// Fetch Data ////////////////////////////

   // Wait for the selected values to be set
   while (!selectedCategory || !selectedName || !selectedVideoId || !selectedFrameId) {
    await new Promise(resolve => setTimeout(resolve, 10));
  }

  console.log('selectedCategory:', selectedCategory, 'selectedName:', selectedName, 'selectedVideoId:', selectedVideoId, 'selectedFrameId:', selectedFrameId);

  ////////////////////////////////// Get Progress Bar
  getProgressBar();

  await fetchCameraParametersAndObjPaths(selectedCategory, selectedName, selectedVideoId, selectedFrameId);
  console.log("Camera Parameters Loaded");
  sceneCanvas = document.getElementById("sceneCanvas");
  imageCanvas = document.getElementById("imageCanvas");
  imageContext = imageCanvas.getContext("2d");

  // Renderer
  renderer = new THREE.WebGLRenderer({ canvas: sceneCanvas });
  renderer.setSize(img_width  * renderResizeRate, img_height  * renderResizeRate);

  sceneCanvas.width = img_width  * renderResizeRate;
  sceneCanvas.height = img_height  * renderResizeRate;
  imageCanvas.width = sceneCanvas.width;
  imageCanvas.height = sceneCanvas.height;

  await setCameraView();

  // Scene
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x91cf93);
  
  console.log('selectedCategory:', selectedCategory, 'selectedName:', selectedName, 'selectedVideoId:', selectedVideoId, 'selectedFrameId:', selectedFrameId);
  const promises = [
    // fetchObjPaths(selectedCategory, selectedName, selectedVideoId, selectedFrameId),
    // loadObjects(),
    
    loadOverlayImage(selectedCategory, selectedName, selectedVideoId, selectedFrameId, selectedUserId),
    displayMasks(selectedCategory, selectedName, selectedVideoId, selectedFrameId, selectedUserId)
];
  await loadObjects();
  
  // await loadOverlayImage(selectedCategory, selectedName, selectedVideoId, selectedFrameId);
  await Promise.all(promises);
  console.log("Overlay Image Loaded");

  const opacitySlider = document.getElementById("opacitySlider");
  opacitySlider.addEventListener("input", updateImageOpacity);

  // const transformationSizeElement = document.getElementById("transformationSize");
  // const transformationSizeValueElement = document.getElementById("transformationSizeValue");
  // transformationSizeElement.addEventListener("input", () => {
  //   console.log("Transformation size:", transformationSizeElement.value);
  //   transformationSize = parseFloat(transformationSizeElement.value);
  //   transformationSizeValueElement.textContent = transformationSizeElement.value;
  // });
  
  // await displayMasks(selectedCategory, selectedName, selectedVideoId, selectedFrameId);
  console.log("Masks Loaded");

  
  
}

function animate() {
  requestAnimationFrame(animate);
  if (scene && camera) {
    renderer.render(scene, camera);
  }
}

//////////////////// Start the Application ////////////////////
await populateDropdowns();
console.log("Dropdowns populated");
init();
animate();