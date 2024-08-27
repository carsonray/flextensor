import numpy as np

class FlexTensor:
    def __init__(self, array, *axes):
        """A multi-dimensional array construct with labeled axes"""
        # Defines tensor values from flextensor or numpy array
        try:
            self.vals = array.vals.copy()
        except AttributeError:
            self.vals = np.array(array)
        
        # Defines axes of tensor
        self.axes = list(axes)

        if len(self.axes) > len(self.shape()):
            # Fills out shape if there are more axes than current shape
            self.vals = self[*[""]*(len(self.axes) - len(self.shape()))]
        else:
            # Appends axis numbers to axes if not all axes are defined
            self.axes = self.axes + [""]*(len(self.shape())-len(self.axes))

    def set_ax(self, axes):
        self.axes = axes
    
    def merge_ax(self, axes):
        # Fills in placeholders with another axis list
        return [other if own == "" else own for own, other in zip(self.axes, axes)]

    def pivot_ax(self, axes):
        return [self.axes[0], axes[1]]
        
    def ax(self, *labels):
        """Determines axis number of label(s)"""
        axes = []
        
        for label in labels:
            if isinstance(label, str):
                try:
                    # Tries finding string label
                    axes.append(self.axes.index(label))
                except ValueError:
                    axes.append(None)
            else:
                # Uses actual index of axis
                axes.append(int(label))

        # If axes has length of one, return first item
        if len(axes) == 1:
            return axes[0]
        else:
            return axes
    
    def __getitem__(self, args):
        """Selects axes of tensor"""
        
        # Sets result array to a copy of tensor values
        result = self.vals.copy()
        
        # Sets num to count axis number
        num = 0

        # Ellipsis position
        ellipsisPos = -1

        # New axes start at the end of the result shape
        newStart = len(result.shape)

        # Map from orginial axis indices to new indices
        axMap = list(range(len(result.shape)))

        # Array to hold source of new axes
        axSource = []

        # Keeps track of axes that have been referenced
        referenced = axMap.copy()
        
        # Loops through arguments
        for arg in args:
            if arg is Ellipsis:
                # If argument is ellipsis, sets position
                ellipsisPos = num
                
            else:
                # If indexing parameter is included
                if isinstance(arg, tuple):
                    # List of slices to be performed (full slices at first)
                    slices = [np.index_exp[:][0]]*(len(result.shape))
                    # Inserts desired slice at axis
                    slices[axMap.index(self.ax(arg[0]))] = arg[1]

                    # Performes slices
                    currLen = len(result.shape)
                    result = result[tuple(slices)]

                    arg = arg[0]
                    # Checks to see if axis was removed
                    if len(result.shape) < currLen:
                        # Clears from referenced list
                        referenced.remove(self.ax(arg))
                        axMap.remove(self.ax(arg))
                        # Decrements new axis start position
                        newStart -= 1
                        # Moves to next axis
                        continue
                    
                
                elif isinstance(arg, list):
                    # Flattens desired axes together into first axis

                    # Flattens arrays into each other backwards
                    for s, t in zip(arg[1:][::-1], arg[:-1][::-1]):
                        source = axMap.index(self.ax(s))
                        target = axMap.index(self.ax(t))
                        # Splits array into blocks along source axis
                        blocks = np.split(result, result.shape[source], axis=source)
                        # Concatenates blocks onto target axis
                        result = np.concatenate(blocks, axis=target)
                        # Reshapes to remove source axis
                        shape = list(result.shape)
                        shape.pop(source)
                        result = result.reshape(shape)
                        # Clears from referenced list
                        referenced.remove(self.ax(s))
                        axMap.remove(self.ax(s))
                        # Decrements new axis start position
                        newStart -= 1

                    arg = arg[0]

                elif arg == "":
                    # If argument is new axis
                    
                    # Creates full slices along length of axes
                    slices = [np.index_exp[:][0]]*(len(result.shape))
                    
                    # Adds axes to array plus the new axis
                    result = result[tuple(slices + [np.newaxis])]

                    # Adds the new axis to the axis order
                    axSource.append(newStart)
                    axMap.append(newStart)
                    
                    # Increments num and newCount
                    num += 1
                    newStart += 1

                    # Moves to next axis
                    continue

                # Argument is axis label
                # Swaps axis specified by label with current position
                axSource.append(self.ax(arg))
                
                # Increments num
                num += 1

                # Removes from reference list
                referenced.remove(self.ax(arg))

        # Creates axis index
        axIndex = np.array([axMap.index(ax) for ax in axSource])

        # Creates reference index
        refIndex = np.array([axMap.index(ax) for ax in referenced])
        
        # Splices in unreferenced axes at ellipsis parameter
        if ellipsisPos > -1 and len(refIndex) > 0:
            axIndex = np.concatenate((axIndex[:ellipsisPos], refIndex, axIndex[ellipsisPos:]))
            
            # Moves num to end of axis list
            num = newStart

        # Transposes result with new axOrder mapped to the sequence of new axes
        result = np.moveaxis(result, source=axIndex, destination=np.arange(num))
        
        # Adds new axes
        axes = self.axes + [""]*(len(result.shape)-len(self.axes))
        
        # Transposes axes
        mixAxes = [axes[axMap[ax]] for ax in axIndex]

        # Returns new flextensor
        return FlexTensor(result, *mixAxes)

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
    
    def __str__(self):
        """Returns string representation"""
        string = f"AxTensor(\n{str(self.vals)}, {self.axes})"
        return string
    

    def __add__(self, other):
        return FlexTensor(self.vals + other.vals, *self.merge_ax(other.axes))
    
    def __sub__(self, other):
        return FlexTensor(self.vals - other.vals, *self.merge_ax(other.axes))
    
    def __mul__(self, other):
        return FlexTensor(self.vals * other.vals, *self.merge_ax(other.axes))
    
    def __truediv__(self, other):
        return FlexTensor(self.vals / other.vals, *self.merge_ax(other.axes))
    
    def __matmul__(self, other):
        return FlexTensor(self.vals @ other.vals, *self.pivot_ax(other.axes))