"""
Most functions will only be a few lines long. 
   See the bottom of the file for how the functions fit together.
  NOTE: In this solution,only the sport column is used for predictions -- events are not used.
   
    WITHOUT SCALING
    k=1: ~31% accuracy
    k=20: ~28% accuracy
    k=500: ~25% accuracy
   WITH SCALING
    k=1: ~30% accuracy
    k=20: ~27% accuracy
    k=500: ~25% accuracy
  SIMPLER (DUMB) METHOD: Always predict 'Athletics', 
       the most common sport. (~38% accuracy!)
"""
import math
import sys
import csv
from collections import Counter

ATHLETES_FILE = './datasets/athletes.csv'

# Original column indices
AGE_COL    = 2
HEIGHT_COL = 3
WEIGHT_COL = 4
GENDER_COL = 5
SPORT_COL  = 12
EVENTS_COL = 13

# Added-on scaled column indices 
#   (assumes events were condensed into a single column)
SCALED_AGE_COL = 14
SCALED_HEIGHT_COL = 15
SCALED_WEIGHT_COL = 16

# These column indices define a point
POINT_COLS = [AGE_COL, HEIGHT_COL, WEIGHT_COL]
SCALED_POINT_COLS = [SCALED_AGE_COL, SCALED_HEIGHT_COL, SCALED_WEIGHT_COL]


def get_input():
    """
    Prompts user for an age, height, and weight.
    """

    age = input("Age (years)? ")
    height = input("Height (cm)? ")
    weight = input("Weight (kg)? ")

    return (age, height, weight)


#########################################
# DO THIS FIRST - k-NN without scaling
#########################################

def load_athletes(filename):
    """
    Loads athlete data from 'filename' into a list of tuples.
    Returns a list of tuples of each athlete's attributes, where
      the last element of each tuple is a list of events the athlete
      competed in.
    The header line is skipped, and rows are removed if missing a value
      for the age, height, or weight.
    For example:
    [...,
     ['Zhaoxu Zhang', "People's Republic of China", 
      '24', '221', '110', 'M', '11/18/1987', 
      '', '0', '0', '0', '0', 
      'Basketball', ["Men's Basketball"]],
     ...
    ]
    """
    assert(type(filename) == str and len(filename) > 0)

    athletes = []

    with open(filename, 'r') as fin:
    #with open(filename, 'r', encoding='utf-8') as fin:

        reader = csv.reader(fin)
        next(reader)                # Skip the header

        # Places all athletes into a list
        athletes = list(reader)

        # Remove rows with empty age/height/weight
        # NOTE: Could alternatively replace empty values with the column mean
        athletes = [row for row in athletes if (all(field != '' for field in row[2:5]))]

        # Place all events in one list in column 'EVENTS_COL'
        athletes = [row[:EVENTS_COL] + [row[EVENTS_COL:]] for row in athletes]

    return athletes


def dist(x, y):
    """ 
    Euclidean distance between vectors x and y. 
    Each element of x and y must be numeric or a numeric string.
    Requires that len(x) == len(y).
    For example: 
        dist((0, 0, 0), (0, 5, 0)) == 5.0
        dist((1, 1, 1), (2, 2, 2)) == 1.7320508075688772
        dist(('1', '1', '1'), ('2', '2', '2')) == 1.7320508075688772
    """

    assert(len(x) == len(y))

    sq_distances = ((float(x[index]) - float(y[index]))**2 for index in range(len(x)))
    
    return math.sqrt(sum(sq_distances))


def nearest_athletes(point, athletes, k = 1):
    """
    Returns the 'k' athletes closest to 'point'.
    Sorts the athletes based on distance to 'point', then return the closest.
    """

    nearest = sorted(athletes, key=lambda athlete: dist(point, athlete[2:5]))

    return nearest[:k]


def most_common_event(athletes):
    """
    Returns the most frequently occuring event in all 'athletes'.
    Consider using Counter.
    """

    # events = Counter(event[] for athlete in athletes for event in athletes[EVENTS_COL])
    # return events.most_common(1)[0][0]
    events_list = []
    
    for event in athletes:
        events_list.append(event[EVENTS_COL][0])
    
    events_count = Counter(events_list)

    return events_count.most_common(1)[0][0]

def most_common_sport(athletes):
    """
    Returns the most frequently occuring sport in all 'athletes'.
    Consider using Counter.
    """

    sports = Counter(athlete[SPORT_COL] for athlete in athletes)

    return sports.most_common(1)[0][0]

############################
# DO THIS SECOND - SCALING
############################

def scale(value, min_value = 0.0, max_value = 1.0):
    """
    For a given value, scales it to the range [0, 1]
       scaled = (value - min_value) / (max_value - min_value)
    Assumes 'min_value' and 'max_value' are floats.
    'value' can be a float or string.
    """
    assert(min_value < max_value)

    return (float(value) - min_value) / (max_value - min_value)


def cols_minmax(data, column_indices):
    """
    Computes the min and max for each column of 'data' in 'column_indices'.
    
    Returns these in the 'mins' and 'maxes' lists. 
        So, mins[0] will be the min of the first column in 'column_indices'.
    For example:
        cols_minmax([[1, 2, 5], [4, 3, 4]], [1,2])
        => ([2,4], [3,5])
    """
    mins = []
    maxes = []

    for col_index in column_indices:
        column = [float(row[col_index]) for row in data]
        mins.append(min(column))
        maxes.append(max(column))

    return (mins, maxes)


def append_scaled_cols(data, column_indices):
    """
    Scales columns in 'data', for each column in 'column_indices'.
    Places the scaled values AT THE END of each row of 'data'.
    Returns the (mins, maxes) of each of the 'column_indices'
    1) Computes the min/max for each column 
    2) Scales each point in the data using the column min/max.
    """
    mins = []
    maxes = []

    # Get mins/maxes for the desired columns
    mins, maxes = cols_minmax(athletes, column_indices)

    # Scale each value
    for row in data:
        for i,col_index in enumerate(column_indices):
            row.append(
                scale(row[col_index], mins[i], maxes[i]))

    return (mins, maxes)


def scale_point(point, mins, maxes):
    """ 
    Scale each element of 'point' using the corresponding
        index of the lists 'mins' and 'maxes'.
    Returns a tuple where each value in the tuple is scaled.
    """

    scaled = (scale(value, mins[i], maxes[i]) for i,value in enumerate(point))

    return tuple(scaled)


############################
# DO THIS THIRD - CROSS-VALIDATION
############################

def cross_validate(athletes, column_indices, k = 20):
    """
    Uses each athlete as a test point. Finds that athlete's nearest neighbors, 
    then sees if the predicted k-NN event matches one of the removed athlete's 
    events. This is an objective measure of classifier performance.
    Returns the percentage accuracy: num_correct / (num_incorrect + num_correct)
    """

    num_correct = 0
    num_incorrect = 0

    #men = [athlete for athlete in athletes if athlete[GENDER_COL] == 'M']
    #women = [athlete for athlete in athletes if athlete[GENDER_COL] == 'F']

    for index, athlete in enumerate(athletes):
        
        # Each test point is an athlete
        test_point = [athlete[col_index] for col_index in POINT_COLS]
        test_point = scale_point(test_point, scale_mins, scale_maxes)
        
        # The 'k+1' is because the closest athlete will be the test point!
        nearest = nearest_athletes(test_point, athletes, k + 1)

#        if athlete[GENDER_COL] == 'M':
#            nearest = nearest_athletes(test_point, men, k + 1)
#        else:
#            nearest = nearest_athletes(test_point, women, k + 1)

        # Find the most common sport of the k-nearest neighbors
        #   Note that nearest[0] IS the athlete.
        predicted_sport = most_common_sport(nearest[1:])

        # Correct if the sport predicted IS the athlete's actual sport!
        if predicted_sport == athlete[SPORT_COL]:
            num_correct += 1
        else:
            num_incorrect += 1

        # Display progress so far every 500 athletes
        if index % 500 == 0:
            print("{} of {}, accuracy so far={}".format( 
                index, len(athletes),
                num_correct / (num_correct + num_incorrect)))

    return num_correct / (num_correct + num_incorrect)


############################
# MAIN PROGRAM
############################

def main:
    ####################
    # GET THE DATA
    ####################
    # Load the athlete data
    athletes = load_athletes(ATHLETES_FILE)
    print 'data is loaded for knn model'

    # # Get the test point and scale it, using the same scale factors
    # test_point = (24, 221, 110)
    test_point = get_input()

    print("YOUR POINT: ", test_point)


    # ####################
    # # SCALE THE POINTS
    # ####################
    # # These scaling routines currently append the 
    # #    exact column values to the end of each row, instead of
    # #    scaling them
     scale_mins, scale_maxes = append_scaled_cols(athletes, POINT_COLS)
     test_point = scale_point(test_point, scale_mins, scale_maxes)

     print("SCALED POINT: ", test_point, "\n")


    # ####################
    # # PERFORM K-NN
    # ####################
    # # Find the nearest athletes to the test_point
    print(athletes)
    nearest = nearest_athletes(test_point, athletes, k=1)

    print("NEAREST ATHLETE(S): ")
    print(nearest)
    print()

    # # Find the most common event of the nearest athletes
    event = most_common_event(nearest)
    print("RECOMMENDED EVENT: ", event, "\n")


    # ####################
    # # CROSS-VALIDATION
    # ####################
    # print("PERFORMING CROSS-VALIDATION ...")
    # # What is the accuracy of this classifier?
    # accuracy = cross_validate(athletes, POINT_COLS, k=1)

    # print("FINAL ACCURACY: ", accuracy)