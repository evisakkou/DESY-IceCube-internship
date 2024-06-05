import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from scipy.optimize import curve_fit

class SignalNeutrinos:
    
    def __init__(self, smax, nsources, spectral_index=-1.9, dnds_index=-2.5, verbose=False):
        self.Smax = smax
        self.nSources = nsources
        self.spectralIndex = spectral_index
        self.dNdSIndex = dnds_index
        
        self.sN = lambda n: self.Smax * n**(1./(self.dNdSIndex + 1.))
        self.Smin = self.sN(nsources)
        
        if verbose:
            print(f"smax: {self.Smax}")
            print(f"nsources: {self.nSources}")
            print(f"Source population:")
            print(f"   Min flux: {self.Smin}")
            print(f"   Max flux: {self.Smax}")
            total_flux = (self.dNdSIndex + 1.) / (self.dNdSIndex + 2.) * self.Smax * (self.nSources**((self.dNdSIndex + 2.) / (self.dNdSIndex + 1.)))
            print(f"   Total flux: {total_flux}")
    
    def createSample(self, lebins=None, invisible_fraction=0., identity_sources=False, use_poisson=True):
        if identity_sources:
            self.srcs = self.nSources * np.ones(self.nSources)
        else:
            self.srcs = self.nSources * np.random.rand(self.nSources)
        
        self.totalFlux = self.sN(self.srcs)
        
        self.isVisible = np.ones(self.nSources)
        if invisible_fraction > 0.:
            self.isVisible = np.where(np.random.rand(self.nSources) < invisible_fraction, 0., 1.)
        
        self.nSourcesVisible = int(self.isVisible.sum())
        print(f"Sources: {self.nSources}, Visible: {self.nSourcesVisible}")

        # Generate isotropic distribution of sources
        self.ra_dist = np.random.uniform(0, 360, self.nSources)
        self.dec_dist = np.arcsin(np.random.uniform(-1, 1, self.nSources)) * 180 / np.pi

        if lebins is not None:
            self.fluxPerBin = np.zeros((self.nSources, len(lebins) - 1))
            self.nEventsPerBin = np.zeros((self.nSources, len(lebins) - 1))
            g1 = self.spectralIndex + 1
            phi0 = g1 * self.totalFlux / (10**(lebins[-1] * g1) - 10**(lebins[0] * g1))
            
            for i, le in enumerate(lebins[:-1]):
                self.fluxPerBin[:, i] = phi0 / g1 * (10**(lebins[i+1] * g1) - 10**(lebins[i] * g1))
                if use_poisson:
                    self.nEventsPerBin[:, i] = np.random.poisson(self.fluxPerBin[:, i]) * self.isVisible
                else:
                    self.nEventsPerBin[:, i] = self.fluxPerBin[:, i] * self.isVisible
                self.fluxPerBin[:, i] *= self.isVisible
            
            self.nEventsTotal = self.nEventsPerBin.sum(axis=1)
           
        else:
            self.fluxPerBin = None
            self.nEventsPerBin = None
            if use_poisson:
                self.nEventsTotal = np.random.poisson(self.totalFlux) * self.isVisible
            else:
                self.nEventsTotal = self.totalFlux * self.isVisible
    
    def plot_ra_dec_distribution(self):
        """Create plot for right ascension and declination"""
        #RA
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(self.ra_dist, bins=50, color='red', alpha=0.5)
        plt.xlabel('RA (degrees)')            
        plt.grid(True)                             
        plt.title('RA Distribution')
        
        #DEC
        plt.subplot(1, 2, 2)
        plt.hist(np.sin(self.dec_dist*np.pi/180), bins=50, color='blue', alpha=0.5)
        plt.xlabel('Dec (degrees)')
        plt.grid(True)                             
        plt.title('Dec Distribution')
        
        plt.tight_layout()
        plt.savefig("RA and DEC for signal")
        plt.show()
    
    def plot_skymap(self, nside=64):
        """Create skymap of sources to check if they are isotropically distributed. We use healpy method. Modify nside number to change the resolution of the map. """
        # Convert RA and Dec to theta and phi for healpy
        theta = np.deg2rad(90 - self.dec_dist)
        phi = np.deg2rad(self.ra_dist)
        
        skymap = np.zeros(hp.nside2npix(nside))
        pixels = hp.ang2pix(nside, theta, phi)
        for pix in pixels:
            skymap[pix] += 1
        
        hp.mollview(skymap, title="Skymap of Sources", cmap="viridis")
        hp.graticule()
        plt.savefig("Skymap Signal Sources")
        plt.show()

    def plot_total_flux(self):
        """Create plot for total flux, using fit curve from scipy. Fit S^-5/2 to check if the plot is correct.
        -Bin Centers: Used as x-values for fitting because they represent the midpoints of the histogram bins.
        -Curve Fit: `curve_fit` function optimizes the value of \( A \) to fit the histogram data."""
        
        plt.figure(figsize=(10,5))
        hist, bins, _= plt.hist(self.totalFlux, bins=50, color= "blue", alpha=0.7, label="Total Flux")
        bin_centers = (bins[:-1] + bins[1:])/2

        #Define S^-5/2 power law with normalisation
        def power_law(x,A):
            return A * x**(-5/2)
        popt,_ = curve_fit(power_law,bin_centers,hist,maxfev=1000)

        plt.plot(bin_centers, power_law(bin_centers,*popt), "r-", label=r'Fit: $S^{-5/2}$')

        plt.xlabel('Total Flux')
        # plt.ylabel('Number of Sources')
        plt.title('Total Flux Distribution with $S^{-5/2}$ fit curve')
        plt.legend()
        plt.grid(True)
        plt.savefig("Total Flux Distribution")
        plt.show()

    def plot_flux_per_energy(self, lebins, normalize=True):
            """Plots the flux per bin with energy from range 3 to 6 GeV"""
            self.createSample(lebins=lebins,invisible_fraction=0.1, identity_sources=False)
            flux_per_energy_bin_random = self.fluxPerBin
            self.createSample(lebins=lebins, invisible_fraction=0.1, identity_sources=True)
            flux_per_energy_bin_identical = self.fluxPerBin
            
            # Summing flux per energy bin for all sources
            total_flux_per_energy_bin_random = flux_per_energy_bin_random.sum(axis=0)
            total_flux_per_energy_bin_identical = flux_per_energy_bin_identical.sum(axis=0)

            if normalize:
                total_flux_per_energy_bin_random /= self.nSources
                total_flux_per_energy_bin_identical /= self.nSources

            energy_bins = 10**lebins
            
            plt.plot(energy_bins[:-1], total_flux_per_energy_bin_random, 'b-', alpha=0.7, label='Random Sources')
            plt.plot(energy_bins[:-1], total_flux_per_energy_bin_identical, 'r-', alpha=0.7, label='Identical Sources')
            
            plt.xlabel('Energy (GeV)')
            plt.ylabel('Total Flux per # of Sources')
            plt.title('Total Flux for Random and Identical Sources')
            plt.xscale('log')
            plt.yscale('log')
            plt.legend()
            plt.show()

    def plot_neutrinos_per_energy(self, lebins):
        # Create samples for random and identical sources
        self.createSample(lebins=lebins, invisible_fraction=0.1, identity_sources=False, use_poisson=True)
        random_with_poisson = self.nEventsPerBin
        self.createSample(lebins=lebins, invisible_fraction=0.1, identity_sources=True, use_poisson=True)
        identical_with_poisson = self.nEventsPerBin
        
        # Create samples without Poisson
        self.createSample(lebins=lebins, invisible_fraction=0.1, identity_sources=False, use_poisson=False)
        random_without_poisson = self.nEventsPerBin
        self.createSample(lebins=lebins, invisible_fraction=0.1, identity_sources=True, use_poisson=False)
        identical_without_poisson = self.nEventsPerBin
        
        # Sum neutrinos per energy bin for all sources
        sum_random_with_poisson = random_with_poisson.sum(axis=0)
        sum_identical_with_poisson = identical_with_poisson.sum(axis=0)
        sum_random_without_poisson = random_without_poisson.sum(axis=0)
        sum_identical_without_poisson = identical_without_poisson.sum(axis=0)
        
        plt.figure(figsize=(10, 6))
        energy_bins = 10**lebins
        
        plt.plot(energy_bins[:-1], sum_random_with_poisson, 'b-', alpha=0.7, label='Random Sources (with Poisson)')
        plt.plot(energy_bins[:-1], sum_identical_with_poisson, 'r-', alpha=0.7, label='Identical Sources (with Poisson)')
        plt.plot(energy_bins[:-1], sum_random_without_poisson, 'g--', alpha=0.7, label='Random Sources (without Poisson)')
        plt.plot(energy_bins[:-1], sum_identical_without_poisson, 'm--', alpha=0.7, label='Identical Sources (without Poisson)')
        
        plt.xlabel('Energy (GeV)')
        plt.ylabel('Number of Neutrinos per Energy Bin')
        plt.title('Number of Neutrinos per Energy Bin for Different Cases')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.show()

lebins = np.log10(np.linspace(3, 6, 6))
signal_random = SignalNeutrinos(smax=100, nsources=500, verbose=True)
signal_random.createSample(lebins, invisible_fraction=0.1, identity_sources=False, use_poisson=True)

#Plots
signal_random.plot_ra_dec_distribution()
signal_random.plot_skymap()
signal_random.plot_total_flux()
signal_random.plot_flux_per_energy(lebins)
signal_random.plot_neutrinos_per_energy(lebins)
