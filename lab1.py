#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

#   Basic code to specify an MDP
#   Learning in Autonomous Systems coursework
#   Aldo Faisal (2015), Imperial College London


class StairClimbingMDP(object):
    def __init__(self):
        # States are:  { s1 <-- s2 <=> s3 <=> s4 <=> s5 <=> s6 --> s7 ]
        self.stateCount = 7
        self.stateNames = ['s1', 's2', 's3', 's4', 's5', 's6', 's7']

        # Actions are: {L,R} --> {0, 1}
        self.actionCount = 2
        self.actionNames = ['L', 'R']

        # Matrix indicating absorbing states
        # P  1   2   3   4   5  6  7  G   <-- STATES
        self.absorbingStates = [1, 0, 0, 0, 0, 0, 1]

        # Load transition
        self.transitionMatrix = self.transitionMatrix()

        # Load reward matrix
        self.rewardMatrix = self.rewardMatrix()

    # get the transition matrix
    def transitionMatrix(self):
        TL = np.zeros((self.stateCount, self.stateCount), dtype=np.int32)
        for i in range(1, self.stateCount - 1):
            TL[i - 1][i] = 1
        TL[0][0] = 1
        TL[self.stateCount - 1][self.stateCount - 1] = 1

        print(TL)

        TR = np.zeros((self.stateCount, self.stateCount), dtype=np.int32)
        for i in range(1, self.stateCount - 1):
            TR[i + 1][i] = 1

        TR[0][0] = 1
        TR[self.stateCount - 1][self.stateCount - 1] = 1

        print(TR)

        transitionMatrix = np.dstack([TL, TR])

        print(transitionMatrix)

        return transitionMatrix  # transition probabilities for each action

    # the transition subfunction
    def transitionFunction(self, priorState, action, postState):
        # Reward function (defined locally)
        prob = self.transitionMatrix[postState, priorState, action]
        return prob

    # get the reward matrix
    def rewardMatrix(self, S=None, A=None):
        # i.e. 11x11 matrix of rewards for being in state s,
        # performing action a and ending in state s'
        if S is None:
            S = self.stateCount
        if A is None:
            A = self.actionCount

        R = np.zeros((S, S, A))

        for i in range(S):
            for j in range(A):
                for k in range(S):
                    R[k, i, j] = self.rewardFunction(i, j, k)

        return R

    # the locally defined reward function
    def rewardFunction(self, priorState, action, postState):
        # reward function (defined locally)
        if ((priorState == 1) and (action == 0) and (postState == 0)):
            rew = -90
        elif ((priorState == 5) and (action == 1) and (postState == 6)):
            rew = 90
        elif (action == 0):
            rew = 10
        else:
            rew = -10

        return rew

    def nextState(self, state, action):
        state = np.argmax(self.transitionMatrix[:, state, action])
        return state

    def valueFunction(self, state, discount=0, policy=[0, 0, 0, 0, 0, 0, 0]):
        maxIterations = 1000  # maximum iterations in case of back and forth
        value = 0
        discFactor = 1
        for i in range(maxIterations):
            # If we are in an absorbing state, terminate
            if(self.absorbingStates[state]):
                break

            # Iteratively add the rewards obtained by the determinitstic policy
            action = policy[state]
            nextState = self.nextState(state, action)
            value += discFactor * self.rewardMatrix[nextState, state, action]

            # Update the discount factor and the current state
            discFactor *= discount
            state = nextState

        return value

    LEFT_POLICY = [0, 0, 0, 0, 0, 0, 0]
    RIGHT_POLICY = [1, 1, 1, 1, 1, 1, 1]

    def plotValues(self, initial_state, policies=[LEFT_POLICY, RIGHT_POLICY]):
        plt.figure()
        for policy in policies:
            gammas = [1/g for g in range(1, 50)]
            values = []

            for gamma in gammas:
                values.append(self.valueFunction(initial_state, gamma, policy))

            plt.plot(gammas, values, label='policy :{}'.format(policy))
        plt.legend()
        plt.show()


st = StairClimbingMDP()
u = st.valueFunction(3, 1, policy=[0, 0, 0, 1, 1, 1, 0])
st.plotValues(3)
