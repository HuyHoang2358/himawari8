import os

def latlon2xy(lat, lon, scale_factor=1.0):
	if lat > 60 or lat < -60 or lon > 205 or lon < 85:
		return -1, -1

	return int((60 - lat) / (0.067682 * scale_factor)), int((lon - 85) / (0.067682 * scale_factor))