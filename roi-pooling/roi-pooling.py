import math

def roi_pool(feature_map: list, rois: list, output_size: int) -> list:
    """
    Returns a list of pooled grids.
    """
    outputs = []
    for x1, y1, x2, y2 in rois:
        region_height = y2 - y1
        region_width = x2 - x1
        pooled = []
        for row in range(output_size):
            row_start = y1 + math.floor(row * region_height / output_size)
            row_end = y1 + math.floor((row + 1) * region_height / output_size)
            row_end = max(row_end, row_start + 1)
            pooled_row = []
            for column in range(output_size):
                column_start = x1 + math.floor(column * region_width / output_size)
                column_end = x1 + math.floor((column + 1) * region_width / output_size)
                column_end = max(column_end, column_start + 1)
                pooled_row.append(max(feature_map[r][c] for r in range(row_start, row_end) for c in range(column_start, column_end)))
            pooled.append(pooled_row)
        outputs.append(pooled)
    return outputs
