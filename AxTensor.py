import numpy as np

class AxTensor:
    def __init__(self, array, *axes):
        """A multi-dimensional array construct with labeled axes"""
        # Defines tensor values from numpy array
        self.vals = np.array(array)
        
        # Defines axes of tensor
        self.axes = list(axes)

        if len(self.axes) > len(self.shape()):
            # Fills out shape if there are more axes than current shape
            self.vals = self.by([len(self.axes) - len(self.shape())])
        else:
            # Appends axis numbers to axes if not all axes are defined
            self.axes = self.axes + list(range(len(self.axes), len(self.shape())))

        
    def ax(self, *labels):
        """Determines axis number of label(s)"""
        axes = []
        
        for label in labels:
            try:
                # Tries finding string label
                axes.append(self.axes.index(label))
            except ValueError:
                # Uses actual index of axis
                axes.append(int(label))

        # If axes has length of one, return first item
        if len(axes) == 1:
            return axes[0]
        else:
            return axes
    
    def by(self, *args):
        """Selects axes of a tensor to return a numpy array"""
        
        # Sets result array to a copy of tensor values
        result = self.vals.copy()
        
        # Sets num to count axis number
        num = 0

        # New axes start at the end of the result shape
        newStart = len(result.shape)

        # List to hold source of new axes
        axSource = np.array([], dtype=int)

        # List to hold destination of new axes
        axDest = np.array([], dtype=int)
        
        # Loops through arguments
        for arg in args:
            if isinstance(arg, list):
                # If argument is newaxis arg specified by
                # [num] where num is number of new axes
                
                # Gets number of new axes
                new = arg[0]
                
                # Creates full slices along length of axes
                slices = [np.index_exp[:][0]]*(len(result.shape))
                
                # Adds axes to array plus the amount of new axes
                result = result[tuple(slices + [np.newaxis]*new)]

                # Adds the new axes to the axis order
                axSource = np.append(axSource, np.arange(newStart, newStart + new))

                # Adds current axis numbers to the axis destination
                axDest = np.append(axDest, np.arange(num, num + new))
                
                # Increments num and newCount by number of new axes
                num += new

                newStart += new
                
            elif arg is Ellipsis:
                # If argument is ellipsis, assume that all axis referencing is complete
                # Sets axis num to the current shape
                num = len(result.shape)
                
            else:
                # Argument is axis label
                
                # Swaps axis specified by label with current position
                axSource = np.append(axSource, [self.ax(arg)])

                # Adds current axis num
                axDest = np.append(axDest, [num])
                
                # Increments num
                num += 1
            
        # Transposes result with new axOrder mapped to the sequence of new axes
        return np.moveaxis(result, source=axSource, destination=axDest)

    def shape(self, *axes):
        """Returns the shape of the tensor along axes"""
        if len(axes) == 0:
            # If no parameters are passed, return full shape of tensor
            return self.vals.shape
        else:
            # Gets shape from each axis
            return tuple([self.vals.shape[self.ax(ax)] for ax in axes])

    def size(self, *axes):
        """Returns the size of the tensor along axes"""
        return np.prod(self.shape(*axes))

    # Magic methods
    def __str__(self):
        """Returns string representation"""
        string = f"AxTensor(\n{str(self.vals)}, {self.axes})"
        return string