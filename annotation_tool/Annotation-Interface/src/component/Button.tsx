import styled from "styled-components"
import { IStyled } from "../type"

interface ButtonProps extends IStyled {
	effectOnClick: () => void
	width?: number
	height?: number
}

const RawButton = (props: ButtonProps) => {
	const { className, effectOnClick, children } = props
	return (
		<button className={className} onClick={effectOnClick}>
			{children}
		</button>
	)
}

const Button = styled(RawButton)`
	width: ${({ width }) => width ?? 40}px;
	height: ${({ height }) => height ?? 40}px;
	background-color: transparent;

	display: flex;
	flex-flow: row nowrap;
	justify-content: center;
	align-items: center;
	cursor: pointer;
`


export const SwitchModeButton = styled.button`
	color: #BF4F74;
	font-size: 1em;
	margin: 1em;
	padding: 0.25em 1em;
	border: 2px solid #BF4F74;
	border-radius: 3px;
	&: hover {
	color: #FFFFFF;
	background-color: #BF4F74;
	}
`;

export const ProcessCoordinatesButton = styled(SwitchModeButton)`
	color: tomato;
	border-color: tomato;
	&: hover {
	color: #FFFFFF;
	background-color: tomato;
	}
`;

export const OtherButton_Red = styled(SwitchModeButton)`
  color: #f44336;
  border-color: #f44336;
  &: hover {
	color: #FFFFFF;
	background-color: #f44336;
	}
`;

//   switch foreground and background color if hover
export const OtherButton_Green = styled(SwitchModeButton)`
  color: #4CAF50;
  border-color: #4CAF50;
  &: hover {
	color: #FFFFFF;
	background-color: #4CAF50;
	}
`;

export const OtherButton_Grey = styled(SwitchModeButton)`
  color: #555555;
  border-color: #555555;
  &: hover {
	color: #FFFFFF;
	background-color: #555555;
	}
`;

export { Button }
