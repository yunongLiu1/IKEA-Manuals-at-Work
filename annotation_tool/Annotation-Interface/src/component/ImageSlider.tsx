import React, { useRef, useState, ChangeEvent } from 'react';
import styled from 'styled-components';
interface ImageSliderProps {
    images: string[];
}

const ImageSlider: React.FC<ImageSliderProps> = ({ images }) => {
    const [sliderValue, setSliderValue] = useState(0);
    const imageContainerRef = useRef<HTMLDivElement>(null);

    const handleSliderChange = (event: ChangeEvent<HTMLInputElement>) => {
        const value = parseInt(event.target.value);
        setSliderValue(value);
        if (imageContainerRef.current) {
            const scrollPosition = value * (imageContainerRef.current.scrollWidth / images.length);
            imageContainerRef.current.scrollLeft = scrollPosition;
        }
    };

    return (
        <div>
            <div ref={imageContainerRef} className="image-slider-image-container">
                {images.map((image, index) => (
                    <img key={index} src={image} alt={`Image ${index}`} />
                ))}
            </div>
            <input
                type="range"
                min="0"
                max="100"
                value={sliderValue}
                onChange={handleSliderChange}
                className="image-slider-slider"
            />
        </div>
    );
};

const StyledImageSlider = styled(ImageSlider)`
.image-slider-image-container {
    display: flex;
    overflow-x: auto;
    scroll-behavior: smooth;
    padding: 10px;
}

.image-slider-image-container img {
    max-height: 200px; // 根据需要调整
    margin-right: 10px;
}

.image-slider-slider {
    width: 100%;
    margin-top: 10px;
}

`;
export default StyledImageSlider;
