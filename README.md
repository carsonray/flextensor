# flextensor
A lightweight python package that allows intuitive axis transposition and axis-based matrix operations in numpy.

*Run the test file to see how it works*

## Features
- Create axes of an n-dimensional array based on identifier strings rather than numbers so you can easily keep track of what is parameterizing your space
- Transpose axes of ndarrays simply by the order of identifiers (or tranditional axis numbers)
- Dymanically create new axes anywhere within the parameter list
- Use ellipsis to skip to the end of the axis parameter list to add new axes or swap old ones
