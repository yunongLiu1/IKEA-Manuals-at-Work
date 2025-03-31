module.exports = {
	root: true,
	parser: "@typescript-eslint/parser",
	plugins: ["@typescript-eslint"],
	extends: [
		"eslint:recommended",
		"plugin:@typescript-eslint/recommended",
		"prettier",
	],
	rules: {
		"@typescript-eslint/no-empty-interface": 1,
		"@typescript-eslint/no-empty-function": 0,
		"@typescript-eslint/no-unused-vars": "warn",
	},
	// turn @typescript-eslint/no-empty-function off




	  
	
}
