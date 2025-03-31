import React from 'react';
import styled from 'styled-components';

// Define an interface for the component's props
interface ProgressBarProps {
  progress: number;
  label: string;
}

const ProgressBarContainer = styled.div`
  width: 100%;
  background-color: #e0e0de;
  border-radius: 10px;
  margin: 10px 0;
`;

// Add type for props
const Filler = styled.div<{ progress: number }>`
  width: ${props => `${props.progress}%`};
  background-color: #4caf50;
  border-radius: inherit;
  text-align: right;
  transition: width 0.5s ease-in-out;
  height: 10px;
`;

const Label = styled.span`
  color: white;
  font-weight: bold;
  height: 10px;
`;

const ProgressBar: React.FC<ProgressBarProps> = ({ progress, label }) => {
  return (
    <ProgressBarContainer>
      <Filler progress={progress}>
        <Label>{`${label}: ${progress.toFixed(0)}%`}</Label>
      </Filler>
    </ProgressBarContainer>
  );
};

export default ProgressBar;
