import sys
import os
import cv2 as cv
import json
import traceback
import numpy as np
from timeit import default_timer as timer
from datetime import timedelta
from concurrent.futures import ProcessPoolExecutor


class PuzzleSolver:
  """
  Given a folder of puzzle piece images, and a source image, Puzzle Solver will find the location of each shuffled puzzle
  peace using computer vision template matching algorithms and then will plot the completed puzzle for inspection
  """
  def __init__(self):
    self.SHOW_ELAPSED_TIME = True
    self.MATCH_SCORE_THRESHOLD = .9
    self.MATCH_ALGORITHMS = [
      'TM_CCORR_NORMED',
      'TM_SQDIFF_NORMED',
      'TM_CCOEFF_NORMED',
    ]
    self.source = cv.imread('source.jpg')
    
    
  def load_puzzle_pieces(self):
    """
    Fetches puzzle piece list data from json file and then each
    associated image is loaded in and saved into puzzle_pieces array
    """
    puzzle_pieces = []
    with open('puzzle_pieces.json', 'r') as json_file:
      pieces = json.load(json_file)
      for piece in pieces:
        template = cv.imread("./puzzle_pieces/%s.jpg" % piece['id'])
        puzzle_pieces.append({
          'id': piece['id'],
          'idx': piece['idx'],
          'template': template,
        })
    return puzzle_pieces
    
    
  def match_template(self, template):
    """
    The function slides the template(puzzle piece) through source image, compares the overlapped patches of size w×h against template
    using the specified method and stores the comparison results in result variable.Each match will have an associated match 'score'.
    If this score doesn't surpass MATCH_SCORE_THRESHOLD, the next algorithm in MATCH_ALGORITHMS will be tried until the threshold is met.
    """
    for method in self.MATCH_ALGORITHMS:
      result = cv.matchTemplate(self.source, template, eval("cv.%s" % method))
      min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
      h, w = template.shape[:2]
      
      # Use min_value for TM_SQDIFF-based algorithms and max_value for the others
      if method.startswith('TM_SQDIFF'):
        score = min_val
        top_left = min_loc
      else:
        score = max_val
        top_left = max_loc
          
      # Calculate Bottom Right, Rect, and XY
      bottom_right = (top_left[0] + w, top_left[1] + h)
      rect = (top_left, bottom_right)
      xy = (int(top_left[0]), int(top_left[1]))
      
      # If Score Threshold Is Not Met, Move To Next Algorithm And Try Again
      if score >= self.MATCH_SCORE_THRESHOLD:
        return score, method, rect, xy
        
    # Raise Exception If No Matches Were Found
    raise Exception(f"Match Template | No Matches Found @ Threshold: {self.MATCH_SCORE_THRESHOLD}")
    
  
  def get_matched_coordinates(self, piece):
    """Match Each Piece And Save Coordinates - Designed For Multi-Process Pool"""
    score, method, rect, xy = self.match_template(piece['template'])
    return (xy, piece['template'])
    
    
  def assemble_puzzle_pieces(self, image_coordinates):
    """Assemble Newly Mapped Puzzle Pieces"""
    completed_puzzle = np.empty_like(self.source)
    h, w = self.source.shape[:2]
    h_step = int(h / 40)
    w_step = int(w / 60)

    # Assemble Puzzle Pieces
    original_pieces = []
    for col in range(0, h, h_step):
      for row in range(0, w, w_step):
        try:
          completed_puzzle[col:col+h_step, row:row+w_step,:] = image_coordinates[(row, col)]
        except:
          print(f'Error Updating Completed Puzzle @ Coordinate: ({row}, {col}). Possibly Missing Puzzle Piece In Folder. Drawing Red Rectangle Around Location.')
          cv.rectangle(completed_puzzle, (row, col), (row+w_step,col+h_step), (0, 0, 255), -1)
    return completed_puzzle
    

  def run(self):
    try:
      print(f'Puzzle Solver Started...')
      
      # Start Timer To Test Improvements From Multiprocessing
      if self.SHOW_ELAPSED_TIME: start = timer()
      
      # Load & Return Puzzle Pieces
      pieces = self.load_puzzle_pieces()
      
      # Multiprocess Pool Executor - Process Coordinate Matching Concurrently - Match Puzzle Pieces & Return Matched Coordinates
      image_coordinates = {}
      with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for r in executor.map(self.get_matched_coordinates, pieces):
          image_coordinates[r[0]] = r[1]
      
      # Assemble Matched Puzzle Pieces And Plot
      completed_puzzle = self.assemble_puzzle_pieces(image_coordinates)
      
      # Showing Completed Puzzle
      cv.imshow('completed_puzzle', completed_puzzle)
      
      # End Timer
      if self.SHOW_ELAPSED_TIME:
        end = timer()
        delta = timedelta(seconds=end-start)
        print(f'\nElapsed Time: {delta - timedelta(microseconds=delta.microseconds)}\n')
        
      cv.waitKey(0)
      cv.destroyAllWindow() 
        
    except Exception as e:
      print(f"Error Solving Puzzle | Reason: {e}\n")
      print(traceback.format_exc())


if __name__ == "__main__":
  puzzle = PuzzleSolver()
  puzzle.run()
  