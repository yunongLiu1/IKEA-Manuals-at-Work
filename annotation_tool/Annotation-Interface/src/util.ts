export const clamp = (n: number, min: number, max: number) => {
	if (n > max) return max
	else if (n < min) return min
	else return n
}

export const rotate = (n: number, min: number, max: number) => {
	if (n > max) return min
	else if (n < min) return max
	else return n
}
