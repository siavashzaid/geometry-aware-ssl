import acoular as ac
from acoupipe.sampler import MicGeomSampler
from acoupipe.datasets.synthetic import DatasetSyntheticConfig
from acoupipe.datasets.features import create_feature

from traits.api import Callable, Int, cached_property, Instance

import numpy as np
from numpy.random import RandomState
from scipy.stats import norm
from scipy.spatial.distance import pdist
from scipy import optimize, special
from scipy.integrate import quad

class CustomMicGeomSampler(MicGeomSampler):
    """
    Microphone geometry sampler with user-defined position generation.

    This sampler delegates the creation of microphone positions to a
    callable function (mpos_fn). This allows flexible, experiment-specific
    microphone array layouts.
    """

    # --- Target microphone geometry object whose positions will be populated ---
    target = Instance(ac.MicGeom, args=())

    # --- Callable that generates microphone positions ---
    mpos_fn = Callable() #mpos_fn(n_mics: int, rng: np.random.Generator) -> np.ndarray

    # --- Maximum and minimum number of microphones in the array
    min_num_mics = Int(1)
    max_num_mics = Int(10)

    # --- Random number generator used for reproducible geometry sampling ---
    generator = Instance(np.random.Generator, args=())

    def _get_mpos_init(self):
        """
        Generate and cache the initial microphone positions.
        """
        if self._mpos_init is None:
            self._mpos_init = self.mpos_fn(
                self.min_num_mics,
                self.max_num_mics,
                self.generator
            )
        return self._mpos_init.copy()

    def sample(self):
        # Reset cached geometry so mpos_fn generates a new one
        self._mpos_init = None
        super().sample()

class VariableArrayConfig(DatasetSyntheticConfig):
    """
    Custom DatasetSynthetic configuration for experiments with variable microphone arrays.

    This config extends AcouPipe's default DatasetSyntheticConfig by:
      1) injecting a custom microphone geometry sampler that uses a user-provided
         position generator (mpos_fn), and
      2) exposing the sampled microphone positions as an explicit dataset feature
         (cartesian_coordinates) so that learning models can directly condition on array geometry.

    """

    # --- Callable that generates microphone positions ---
    mpos_fn = Callable() #mpos_fn(n_mics: int, rng: np.random.Generator) -> np.ndarray

    # --- Random number generator used for reproducible geometry sampling ---
    generator = Instance(np.random.Generator, args=())

    min_num_mics = Int(1)
    
    max_num_mics = Int(10)

    def create_micgeom_sampler(self):
        """
        Constructs the microphone geometry sampler using the user-defined position generator.

        Returns
        -------
        CustomMicGeomSampler
            Sampler that generates a microphone array geometry using mpos_fn.
        """
        return CustomMicGeomSampler(
            #No additional random perturbation applied here (scale=0).
            random_var=norm(loc=0, scale=0),
            ddir=np.array([[1.0], [1.0], [1.0]]),
            target=self.noisy_mics,

            #Variable number of microphones, but currently fixed at 5 for testing. 
            #Adjust min/max as needed.
            min_num_mics=self.min_num_mics,
            max_num_mics=self.max_num_mics,

            #User-defined microphone position function + RNG.
            mpos_fn=self.mpos_fn,
            generator=self.generator,
        )

    def _get_mdim(self):
        """
        Override microphone dimension handling.
        """
        return None

    def _get_default_feature_cartesian_coordinates(self, **kwargs):
        """
        Exposes the sampled microphone coordinates as a dataset feature so
        learning models can directly condition on array geometry.

        Returns
        -------
        Feature
            A feature named 'cartesian_coordinates' with shape (3, N),
            where N is the number of microphones in the sampled geometry.
        """
        def feat_func(sampler):
            pos = sampler[1].target.pos  # shape: (3, N)
            return {"cartesian_coordinates": pos}

        return create_feature(
            feat_func,
            name="cartesian_coordinates",
            dtype=float,
            shape=(3, None),
        )

def random_positions(min_num_mics, max_num_mics, generator):
    """
    Generate a random centered planar microphone array with normalized aperture.

    Parameters
    ----------
    min_num_mics : int
        Minimum number of microphones to sample.
    max_num_mics : int
        Maximum number of microphones to sample.
    generator : np.random.Generator
        NumPy random number generator for reproducible sampling.

    Returns
    -------
    np.ndarray
        Microphone positions with shape (3, N), where N is the number of
        microphones.
    """

    # --- sample number of microphones ---
    n = generator.integers(min_num_mics, max_num_mics + 1)

    # --- sample positions uniformly over a disk ---
    # sqrt ensures uniform spatial density in polar coordinates
    r = 0.5 * np.sqrt(generator.random(n))
    theta = 2 * np.pi * generator.random(n)

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.zeros(n)

    # --- assemble planar coordinates ---
    positions = np.column_stack((x, y, z))  # shape: (N, 3)

    # --- center array at the origin (translation invariance) ---
    positions -= positions.mean(axis=0)

    # --- normalize array aperture (scale invariance) ---
    # max pairwise distance equals 1 after normalization
    positions /= np.max(pdist(positions))

    # --- return in AcouPipe-compatible shape (3, N) ---
    return positions.T

def VogelHansen(min_num_mics, max_num_mics, generator):

    #uniformly distribute H and M
    M = generator.integers(min_num_mics, max_num_mics + 1) 
    H = generator.uniform(-1,4)

    r_max=0.5
    V=5

    def F(r):
        if H<0:
            return 1/special.iv(0,np.pi*H*np.sqrt(1+0j-r*r))
        
        return special.iv(0,np.pi*H*np.sqrt(1+0j-r*r))
    
    def Freal(r):
        return F(r).real
    
    FI = quad(Freal,0,1)[0]  

    def Froot(r):
        A = FI/(M*F(r))
        r0 = np.sqrt(np.cumsum(A)).real
        return (r-r0)
    
    rz = optimize.leastsq(Froot,np.zeros(M)/M+0.01)
    rz = rz[0]
    rm = rz * r_max/rz.max() 
    n = np.arange(M)+1
    theta = np.pi*n*(1+np.sqrt(V))
    xyz = np.zeros((3,M),dtype=np.double)
    xyz[0] = rm*np.cos(theta)
    xyz[1] = rm*np.sin(theta)

    # Centering
    xyz[0] -= np.mean(xyz[0])
    xyz[1] -= np.mean(xyz[1])

    #Normalizing 
    aperture = np.max(pdist(xyz[:2].T)) #Aperture in acoupipe is defined as the maximum pairwise distance between microphones
    xyz[:2] /= aperture

    return xyz