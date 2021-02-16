import numpy as np


class SingleViewM:
    def __init__(self, img: np.ndarray, endpoints: dict, refpoints: dict):
        assert sorted(endpoints.keys()) == ["x", "y",
                                            "z"], "The endpoints dictionary should only contain the keys 'x', 'y' and 'z' and they must all be present"
        assert sorted(refpoints.keys()) == ["o", "x", "y",
                                            "z"], "The reference points dictionary should only contain the keys 'x', 'y', 'z' and 'o' and they must all be present"

        for key in endpoints:
            assert len(
                endpoints[key]) == 3, f"You should provide 3 lines, instead, you have {len(endpoints[key])} lines"
        self.endpoints = endpoints
        self.refpoints = refpoints
        self.img = img
        self.vanishing_points = {}

    def findVanishingPoints(self) -> dict:
        """
            Calculate vanishing points from a dictionary of endpoints using Bob Collin's method.
        """
        if not self.vanishing_points:
            for key in self.endpoints:
                lines = []
                # Find line equation from every pair of endpoints
                for endpoints in self.endpoints[key]:
                    e1, e2 = endpoints
                    # Homogeneous coordinates
                    e1 = list(e1) + [1]
                    e2 = list(e2) + [1]
                    lines.append(np.cross(e1, e2))
                M = np.zeros((3, 3), dtype='float64')
                for i in range(3):
                    a, b, c = lines[i]
                    M += np.array([[a * a, a * b, a * c], [a * b, b * b, b * c], [a * c, b * c, c * c]])
                # Compute vanishing points
                eig_values, eig_vectors = np.linalg.eig(M)
                vanishing = eig_vectors[:, np.argmin(eig_values)]
                vanishing = vanishing / vanishing[-1]
                self.vanishing_points[key] = vanishing

        return self.vanishing_points

    def findProjectionMatrix(self) -> np.ndarray:
        if not self.vanishing_points:
            self.findVanishingPoints()
        # We first calculate the scaled version of the projection matrix
        proj_matrix = np.stack(
            [self.vanishing_points['x'], self.vanishing_points['y'], self.vanishing_points['z'],
             self.refpoints['o']],
            axis=1)
        # Now we'll find the scales associated to x, y and z
        scales = {}
        for key in self.vanishing_points:
            ref_length = np.linalg.norm(self.refpoints[key] - self.refpoints['o'].reshape(-1, 1))
            a, _, _, _ = np.linalg.lstsq(self.vanishing_points[key].reshape(-1, 1) - self.refpoints[key],
                                         (self.refpoints[key] - self.refpoints['o'].reshape(-1, 1)))
            scales[key] = a[0, 0] / ref_length
        # Use the scales to get the correct matrix
        proj_matrix[:, 0] = proj_matrix[:, 0] * scales['x']
        proj_matrix[:, 1] = proj_matrix[:, 1] * scales['y']
        proj_matrix[:, 2] = proj_matrix[:, 2] * scales['z']
        return proj_matrix

    def generateVRML(self, path: str):
        return NotImplementedError

    def generateTextures(self, proj_matrix: np.ndarray):
        H_xy = proj_matrix[:, [0, 1, 3]].copy()
        H_yz = -proj_matrix[:, [1, 2, 3]].copy()
        H_xz = -proj_matrix[:, [0, 2, 3]].copy()
