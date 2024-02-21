import numpy as np


def find_nearest(courses, case):
    newCourses = []
    for course in courses:
        gradient = (course[-1][1] - course[0][1]) / (course[-1][0] - course[0][0])
        x = np.linspace(0, 1000, len(case))
        y = gradient * x
        newCourses.append(np.array([x, y]).T)
    # add empty array to store the differences with numpy
    diff = np.array([])
    for n_course in newCourses:
        subtract_func = np.abs(n_course[:, 1] - case[:, 1])  # abs of grape gap
        diff = np.append(diff, np.sum(subtract_func) / len(case))  # append the subtract_func to diff
    return np.min(diff)
