"""
This file implements the Extended Kalman Filter.
"""

import numpy as np

from filters.localization_filter import LocalizationFilter
from tools.task import get_motion_noise_covariance
from tools.task import get_observation as get_expected_observation
from tools.task import get_prediction
from tools.task import wrap_angle
from tools.task import get_landm

class EKF(LocalizationFilter):
	
	def G_t(prev_state,ut):
		
    def predict(self, u):
        # TODO Implement here the EKF, perdiction part. HINT: use the auxiliary functions imported above from tools.task
		x=get_prediction(self._state,u)
		
		Gt=np.array([[1,0,-u[1]*np.sin(x[2]+u[0])],[0,1,u[1]*np.cos(x[2]+u[0])],[0,0,1]])
        M=get_motion_noise_covariance(u,self._alphas)
        Vt=np.array([[-u[0]*np.sin(x[2]+u[0]),cos(x[2]+u[0]),0],[u[0]*cos(x[2]+u[0]),sin(x[2]+u[0]),0],[1,0,,1]])
		
		self.mu=self._state.mu+np.array([u[1]*np.cos(x[2]+u[0]),u[1]*np.sin(x[2]+u[0]) ,u[0]+u[2]])
		
		self.Sigma=G_t @ self.Sigma @ G_t.T+ Vt @ M @ Vt.T
        self._state_bar.mu = self.mu[np.newaxis].T
        self._state_bar.Sigma = self.Sigma
		
    def update(self, z):
        # TODO implement correction step
		
		observ_z=get_observation(self._state,z[1])
		dx=get_landm(self._state,z[1])
		Ht=np.array([dx[1]/(dx[1]**2 + dx[0]**2),-dx[0]/(dx[1]**2 + dx[0]**2),-1])
		St=Ht @ self._state_bar.Sigma @ Ht.T +self._Q
		K=self._state_bar.Sigma @ Ht.T/St
		self._state_bar.mu=self._state_bar.mu+K*(observ_z[0]-z[0])
		self._state_bar.Sigma=(np.eye(3)-K@ Ht)@ self._state_bar.Sigma
        self._state.mu = self._state_bar.mu
        self._state.Sigma = self._state_bar.Sigma
