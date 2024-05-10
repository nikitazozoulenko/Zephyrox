import unittest

import numpy as np
import jax
import jax.numpy as jnp
import iisignature

from features.sig_trp import SigVanillaTensorizedRandProj

N=2
T=10
D=3
prng_key, trp_key = jax.random.split(jax.random.PRNGKey(0))

class TestSigVanillaTensorizedRandProj(unittest.TestCase):

    def test_approx_against_iisignature_trunc_level(self, trunc_level):
        # test specfic arguments
        n_features = 100000
        max_batch = N
        concat_levels=False

        #input brownian motions
        X, Y = jnp.cumsum(jax.random.normal(prng_key, (2, N, T, D)), axis=2)

        #compare
        trp = SigVanillaTensorizedRandProj(trp_key, n_features, trunc_level, max_batch, concat_levels).fit(X)
        X_out_trp = trp.transform(X)
        Y_out_trp = trp.transform(Y)
        dot_trp = jnp.dot(X_out_trp, Y_out_trp.T)

        X_out_sig = iisignature.sig(np.array(X), trunc_level)
        Y_out_sig = iisignature.sig(np.array(Y), trunc_level)
        dot_sig = np.dot(X_out_sig, Y_out_sig.T)

        print(dot_trp)
        print(dot_sig)
        print("\n")
        self.assertTrue(np.allclose(np.array(dot_trp), dot_sig, atol=1e-3))



    def test_approx_against_iisignature(self):
        for i in range(1, 7):
            self.test_approx_against_iisignature_trunc_level(i)


# python -m unittest -v tests/features/test_sig_trp.py 
if __name__ == '__main__':
    unittest.main()