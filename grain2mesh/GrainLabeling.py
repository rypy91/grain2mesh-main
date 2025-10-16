"""
© 2025. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

Class definition for GrainLabeling

Functionality:
- Relabel regions
- Label grains based on area
- Remove floating grain spaces
- Merge small regions
"""
import numpy as np
from skimage import morphology, measure
from skimage.morphology import dilation, footprint_rectangle



class GrainLabeling:
    def __init__(self, labels):
        self.labels = labels    # section IDs
        self.grain_sizes = None # section areas

    def label_zero_spaces(self):
        """
        Assign unique negative label to each discrete 0 region
        """
        # Identify regions of zeros (ie. pore space)
        zero_regions = self.labels == 0 

        # Uniquely lable each zero space
        zero_labels = measure.label(zero_regions, connectivity=1)

        # Give each pore space a unique negative label, grain spaces all labeled 0
        negative_labels = np.where(zero_labels > 0, -zero_labels, 0)

        # Update labels with negative values for pore spaces
        self.labels[zero_regions] = negative_labels[zero_regions]

    def fix_diagonal_pores(self):
        """
        Resolve diagonal pore-to-pore 'corner' connections by reassigning those pore
        pixels to adjacent grain labels.

        A corner case is when (x,y) and (x+dx,y+dy) are pores (<= 0), while the two
        orthogonal neighbors along that corner (x+dx,y) and (x,y+dy) are grains (> 0).
        This method finds all such cases for (dx,dy) in {(1,1),(1,-1),(-1,1),(-1,-1)}
        and reassigns BOTH pore pixels to a nearby grain label.
        """
        lbl = self.labels
        n, m = lbl.shape

        # Treat <= 0 as pore
        is_pore = lbl <= 0

        # Precompute global areas for positive labels (for tie-breaking)
        flat = lbl.ravel()
        pos = flat > 0
        if pos.any():
            max_lab = int(flat.max())
            counts = np.bincount(flat[pos])
            # np.bincount starts at label 1; align indices 1..max_lab
            # For labels that don't appear, area = 0.
            def label_area(L):
                if L <= 0 or L > counts.size:
                    return 0
                return counts[L - 1]
        else:
            # no grains? nothing to do
            return

        # Collect updates without mutating during scan
        to_update_coords = []
        to_update_labels = []

        # Four diagonal directions and their paired orthogonals
        corners = [
            ( 1,  1),  # checks (x+1,y) and (x,y+1)
            ( 1, -1),  # checks (x+1,y) and (x,y-1)
            (-1,  1),  # checks (x-1,y) and (x,y+1)
            (-1, -1),  # checks (x-1,y) and (x,y-1)
        ]

        for dx, dy in corners:
            # Valid interior range so (x+dx,y+dy), (x+dx,y), (x,y+dy) are in bounds
            x0 = max(0, -dx)
            x1 = min(n, n - dx)
            y0 = max(0, -dy)
            y1 = min(m, m - dy)

            # Slices
            P00 = is_pore[x0:x1, y0:y1]                   # (x,y)
            P11 = is_pore[x0+dx:x1+dx, y0+dy:y1+dy]       # (x+dx,y+dy) diag
            G10 = lbl[x0+dx:x1+dx, y0:y1]                 # (x+dx,y)
            G01 = lbl[x0:x1, y0+dy:y1+dy]                 # (x,y+dy)

            # Corner condition: both pores on the diagonal pair, and both orthogonals are grains (>0)
            mask_corner = P00 & P11 & (G10 > 0) & (G01 > 0)
            if not np.any(mask_corner):
                continue

            # Coordinates within the sliced window where the condition holds
            xs, ys = np.nonzero(mask_corner)

            for k in range(xs.size):
                x = xs[k] + x0
                y = ys[k] + y0

                # Determine the two orthogonal grain labels for this corner
                lab1 = lbl[x + dx, y]       # (x+dx,y)
                lab2 = lbl[x, y + dy]       # (x, y+dy)

                # Pick winner: majority (if equal, use area tie-break; else first)
                if lab1 == lab2:
                    winner = lab1
                else:
                    # equal vote -> use global area
                    a1 = label_area(int(lab1))
                    a2 = label_area(int(lab2))
                    if a1 > a2:
                        winner = lab1
                    elif a2 > a1:
                        winner = lab2
                    else:
                        winner = lab1  # deterministic fallback

                # Reassign BOTH pore pixels of the diagonal pair to the winner
                p1 = (x, y)
                p2 = (x + dx, y + dy)

                # Only queue updates if still pore at those locations (avoid duplicates)
                if lbl[p1] <= 0:
                    to_update_coords.append(p1)
                    to_update_labels.append(winner)
                if lbl[p2] <= 0:
                    to_update_coords.append(p2)
                    to_update_labels.append(winner)

        if to_update_coords:
            xs, ys = zip(*to_update_coords)
            xs = np.array(xs)
            ys = np.array(ys)
            self.labels[xs, ys] = np.array(to_update_labels, dtype=lbl.dtype)

    def assign_grain_sizes(self):
        """ Measure area of each discrete region """
        properties = measure.regionprops(self.labels)
        self.grain_sizes = np.zeros_like(self.labels, dtype=np.float32)

        # For each label, assign grain size based on area
        for prop in properties:
            if prop.label > 0:
                self.grain_sizes[self.labels == prop.label] = prop.area

    def remove_floaters(self):
        """ Remove regions adjacent only to pore space (floaters) """
        adjacencies = self._find_adjacent_regions()
        for key, value_set in adjacencies.items():
            if all(x < 0 for x in value_set):
                print('Found a floater. Removing now.')
                self.labels[self.labels == key] = list(value_set)[0]

    def _find_adjacent_regions(self):
        """ Create a dictionary of adjacent regions """
        unique_labels = np.unique(self.labels)
        adjacencies = {label: set() for label in unique_labels} # map of adjacent regions

        for label in unique_labels:
            if label == 0:
                continue
            
            mask = self.labels == label # region with current label
            dilated = dilation(mask, footprint_rectangle((3, 3))) # region expanded by 1 pixel in all directions

            neighbors = np.unique(self.labels[dilated & ~mask]) # inverse of mask; just neighbors
            for n in neighbors:
                if n != label:
                    adjacencies[label].add(n) # track adjacent pixels not in label/region

        return adjacencies

    def remap_labels(self, start_from_zero):
        """
        Shift labels to be consecutive starting at either 0 or 1

        Args:
        - start_from_zero (bool): flag for whether to start at 0 or 1
        """
        unique_values = np.unique(self.labels)
        new_values = None
        if start_from_zero:
            new_values = np.arange(len(unique_values)) # shift labels so now starting at 0
        else:
            new_values = np.arange(1, len(unique_values) + 1) # shift labels starting at 1
        
        self.value_mapping = {old_val: new_val for old_val, new_val in zip(unique_values, new_values)}
        # reorder lables based on mapping: old value = key, new value = value
        self.labels = np.vectorize(self.value_mapping.get)(self.labels)

    def merge_small_regions(self, threshold):
        """
        Merges small regions into adjacent regions

        Args:
        - threshold (float): minimum area for a region
        """
        n=np.size(self.labels,0)
        m=np.size(self.labels,1)
        self.remap_labels(False)
        
        properties = measure.regionprops(self.labels)
        # adjacencies = self._find_adjacent_regions()
        for region in properties:
            if region.area < threshold:
                print(f'Merging small region: {region.label}')

                # Find the adjacent regions
                adjacencies = self._find_adjacent_regions()
                adj_regions = list(adjacencies[region.label])
  
                small_region_mask = (self.labels == region.label)
                small_centroid = properties[region.label - 1].centroid
                small_region_coords = np.argwhere(small_region_mask)

                #Initialize variable to store the new label
                new_label = None

                #special case (single pixel..) #special case contributed by Eban Bozek beta tester
                if region.area == 1.0:
                    coords=np.where(self.labels == region.label)
                    x=coords[0][0]
                    y=coords[1][0]
                    COORDS=np.array([x+1,x-1,y-1,y+1])
                    
                    # Make sure coords stay within image bounds
                    COORDS[COORDS<0]=0
                    if (x+1)>=n:
                        COORDS[0]=n-1
                    if (y+1)>=m:
                        COORDS[3]=m-1
                    # list of adjacent segment labels: [up,down,right,left]
                    TEMP=[self.labels[x,COORDS[3]] ,self.labels[x,COORDS[2]],self.labels[COORDS[0],y],self.labels[COORDS[1],y] ]
                    if TEMP.count(int(0))>0:
                        TEMP=list(filter(lambda a: a!=0,TEMP))

                    else:
                        
                        # find the unique adjacent segment labels, and the respective counts
                        [LABELS,COUNTS]=np.unique(TEMP,return_counts=True) 
                        
                        if len(LABELS)>1: # if there is more than one nearby grain
                            [LABELS2,COUNTS2]=np.unique(COUNTS,return_counts=True) # find if number of counts is repeated
                            
                            if len(LABELS2)==1: #if all counts are equal (one of two/three segments, or two of two segments)
                                
                                label=LABELS[0]
                                
                            else:# one label was found more than the others
                                M=np.argmax(COUNTS)
                                label=LABELS[M]
                               
                        else: #only one nearby grain segment 
                            label=LABELS[0]
                    #assign found label to this pixel, or zero if skipped
                    self.labels[x,y]=label
                    
                    #self.labels[self.labels == region.label] = adj_regions[0]

                
                # Iterate over each adjacent region to find if the centroid is within its bounding box
                for adj_label in adj_regions:
                    # Get the properties of the adjacent region
                    adj_region_props = properties[adj_label - 1]
                    adj_bbox = adj_region_props.bbox  # Bounding box of the adjacent region
                    
                    # Check if the centroid is within the bounding box of the adjacent region
                    if self._is_point_within_bbox(small_centroid, adj_bbox):
                        new_label = adj_label
                        break  # Exit loop once the matching adjacent region is found
                
                if new_label is not None:
                    # If centroid is within bounding box, update the label
                    self.labels[self.labels == region.label] = new_label
                else:
                    # adjacencies = self._find_adjacent_regions()
                    # adj_regions = list(adjacencies[region.label])
                    # Create a distance map to find the closest adjacent region for each pixel
                    distance_map = np.full(small_region_mask.shape, np.inf)
                    for adj_label in adj_regions:
                        # Get the binary mask of the adjacent region
                        adj_region_mask = (self.labels == adj_label)
                        adj_region_coords = np.argwhere(adj_region_mask)

                        # Calculate distance from each pixel in the small region to the current adjacent region
                        for pixel in small_region_coords:
                            distances = np.sqrt(np.sum((adj_region_coords - pixel) ** 2, axis=1))
                            min_distance = np.min(distances)
                            if min_distance < distance_map[tuple(pixel)]:
                                distance_map[tuple(pixel)] = min_distance
                                self.labels[tuple(pixel)] = adj_label

#NOTE there is known bug with having pore space be so big that it wraps back around and creates an of diagonal []^[] connection.. see commented out
#see line 116 in cubitSurface for debugging purposes..

    def _is_point_within_bbox(self, point, bbox):
        """
        Determine if a point is within a given bounding box

        Args:
        - point (tuple): 2D point coordinates
        - bbox (tuple): bounding box

        Returns:
        - bool:
        """
        x, y = point
        min_row, min_col, max_row, max_col = bbox
        return min_row <= x < max_row and min_col <= y < max_col

    def set_negative_labels_to_zero(self):
        """ Combine negative labels into a single region """
        for key, value in self.value_mapping.items():
            if key >= 0:
                break
        pp_thresh = value - 1

        # Label all negative spaces 0 (pore spaces)
        self.labels[self.labels <= pp_thresh] = 0

    # def label_by_phase(self, labels, original):
    #     """"""
    #     mapped_labels = np.zeros_like(labels)

    #     for label in np.unique(labels):
    #         if label == 0:
    #             continue

    #         mask = labels == label
    #         original_labels = original[mask]

    #         if original_labels.size > 0:
    #             majority_label = np.bincount(original_labels.astype(np.uint8)).argmax()
    #             mapped_labels[mask] = majority_label

    #     return mapped_labels
