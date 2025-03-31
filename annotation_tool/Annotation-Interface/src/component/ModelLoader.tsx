import React, { useEffect } from 'react';
import * as THREE from 'three';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader';
import { useLoader } from '@react-three/fiber';

// Model component that takes the model file path as a prop
function Model({ filePath, highlightColors, ...props }: { filePath: string, highlightColors?: string } & JSX.IntrinsicElements['group']) {
  const obj = useLoader(OBJLoader, filePath);

  // Inspecting materials after the model is loaded
  useEffect(() => {
    obj.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        console.log("Material of the mesh:", child.material);

        // // Modify the material to render both sides of the faces
        // child.material.side = THREE.DoubleSide; //Move to below
        // Some materials cannot be displayed on both sides



        if (!child.material) {
          console.error('Material is undefined');
          return;
        }
        if (Array.isArray(child.material)) {
          
          // Handle array of materials
          child.material.forEach((mat) => {
            mat.side = THREE.DoubleSide;

            if (!mat.color) {
              console.error('Color is undefined');
              return;
            }
            // Now it should be safe to set the color
            if (highlightColors) {
              mat.color.set(highlightColors);
            }
          });
        } else {
          child.material.side = THREE.DoubleSide;
          if (!child.material.color) {
            console.error('Color is undefined');
            return;
          }
          // Now it should be safe to set the color
          if (highlightColors) {
            child.material.color.set(highlightColors);
          }
        }
      }
    });
  }, [obj, highlightColors]);

  return (
    <group {...props} dispose={null}>
      <primitive object={obj} />
    </group>
  );
}

// Component that takes an array of file paths and creates Model components
export function ModelLoader({ filePaths, highlightColors }: { filePaths: string[]; highlightColors: Record<string, string> }) {
  return (
    <>
      {filePaths.map((filePath, index) => (
        <Model key={index} filePath={filePath} highlightColors={highlightColors[filePath]}  />
      ))}
    </>
  );
}



