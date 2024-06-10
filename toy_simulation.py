import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from scipy.optimize import curve_fit


class BackgroundNeutrinos:
    def __init__(self):
        self.ra_min = 0 
        self.ra_max = 360 
        self.dec_min = -90 
        self.dec_max = 90 

    def power_law_distribution(self, energy_min, energy_max, size, alpha):
        u = np.random.uniform(0, 1, size)
        energies = (energy_max**(alpha+1) - energy_min**(alpha+1)) * u + energy_min**(alpha+1)
        energies = energies**(1 / (alpha+1))
        return energies

    def simulate_events(self, n_sources, energy_range=(1e12, 1e15)):
        # Generate 50,000 events for the energy distribution with spectral index -3.0
        bkg_sources = 50000
        ra_dist1 = np.random.uniform(0, 360, bkg_sources)
        dec_dist1 = np.arcsin(np.random.uniform(-1, 1, bkg_sources)) * 180 / np.pi
        energies_dist1 = self.power_law_distribution(*energy_range, bkg_sources, -3.0)
    
        # Generate 500 events for the energy distribution with spectral index -1.9
        sig_sources = 500
        ra_dist2 = np.random.uniform(self.ra_min, self.ra_max, sig_sources)
        dec_dist2 = np.arcsin(np.random.uniform(-1, 1, sig_sources)) * 180 / np.pi
        energies_dist2 = self.power_law_distribution(*energy_range, sig_sources, -1.9)

        return {
            'atm bkg': (ra_dist1, dec_dist1, energies_dist1),
            'astr bkg': (ra_dist2, dec_dist2, energies_dist2)
        }

    def plot_histograms(self, events):
        fig, ax = plt.subplots(2, 3, figsize=(18, 10))
        fig.subplots_adjust(hspace=0.3)

        for i, (event_type, (ra, dec, energies)) in enumerate(events.items()):
            ra = np.array(ra)
            dec = np.array(dec)
            energies = np.array(energies)
            
            # Plot histogram for RA
            ax[0, i].hist(ra, bins=60, color='blue', alpha=0.5)
            ax[0, i].set_xlabel('Right Ascension (RA)')
            # ax[0, i].set_ylabel('Number of Sources')
            ax[0, i].set_title(f'Histogram of Right Ascension (RA) for {event_type.capitalize()}')
            ax[0, i].grid(True)
            
            # Plot histogram for Dec
            ax[1, i].hist(np.sin(dec*np.pi/180), bins=60, color='green', alpha=0.5)
            ax[1, i].set_xlabel('Declination (Dec)')
            # ax[1, i].set_ylabel('Number of Sources')
            ax[1, i].set_title(f'Histogram of Declination for {event_type.capitalize()}')
            ax[1, i].grid(True)

            # Plot histogram for energies
            ax[i, 2].hist(np.log10(energies), bins=60, color='red', alpha=0.5, label=f'{event_type.capitalize()}')
            ax[i, 2].set_xlabel('log10(Energy)')
            # ax[i, 2].set_ylabel('Number of Sources')
            ax[i, 2].set_title('Histogram of Energy')
            ax[i, 2].semilogy()
            ax[i, 2].grid(True)
            ax[i, 2].legend()

        fig.savefig('Histograms_Background.png')

        # Combined histogram for energies
        plt.figure(figsize=(10, 5))
        plt.hist([np.log10(events['atm bkg'][2]), np.log10(events['astr bkg'][2])], bins=60, color=['blue', 'red'], alpha=0.5, label=['Atm Bkg', 'Astr Bkg'])
        plt.xlabel('log10(Energy)')
        # plt.ylabel('Number of Sources')
        plt.title('Combined Histogram of Background Energy')
        plt.semilogy()
        plt.grid(True)
        plt.legend()
        plt.savefig("Background_Energy.png")
        plt.show()

    def plot_skymap(self, events, nside=64):
        for event_type, (ra, dec, energies) in events.items():
            # Convert RA and Dec to theta and phi for healpy
            theta = np.deg2rad(90 - dec)
            phi = np.deg2rad(ra)
            
            # Create a healpy map
            skymap = np.zeros(hp.nside2npix(nside))
            pixels = hp.ang2pix(nside, theta, phi)
            for pix in pixels:
                skymap[pix] += 1
            
            # Plot the healpy map
            hp.mollview(skymap, title=f"Skymap of {event_type.capitalize()} Sources for background", cmap="viridis")
            hp.graticule()
            plt.show()

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
            self.fluxPerBin=np.zeros((self.nSources,len(lebins)-1)) 
            self.nEventsPerBin=np.zeros((self.nSources,len(lebins)-1)) 
            g1=self.spectralIndex+1
            phi0=g1*self.totalFlux/( 10**(lebins[-1]*g1) - 10**(lebins[0]*g1) )
            
            for i,le in enumerate(lebins[:-1]):
                self.fluxPerBin[:,i]=phi0/g1*(10**(lebins[i+1]*g1) - 10**(lebins[i]*g1))
                self.nEventsPerBin[:,i]=np.random.poisson(self.fluxPerBin[:,i])*self.isVisible #i can change this to if poisson. Make poisson default and if else we keep it the same meaning we just keep flux per bin
                self.fluxPerBin[:,i]*=self.isVisible
                
            self.nEventsTotal=self.nEventsPerBin.sum(1)
           
        else:
            self.fluxPerBin=None
            self.nEventsPerBin=None
            self.nEventsTotal=np.random.poisson(self.totalFlux)*self.isVisible
    
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
        
        # Sum neutrinos per energy bin for all sources
        sum_random_with_poisson = random_with_poisson.sum(axis=0)
        sum_identical_with_poisson = identical_with_poisson.sum(axis=0)
        
        plt.figure(figsize=(10, 6))
        energy_bins = 10**lebins
        
        plt.plot(energy_bins[:-1], sum_random_with_poisson, 'b-', alpha=0.7, label='Random Sources')
        plt.plot(energy_bins[:-1], sum_identical_with_poisson, 'r-', alpha=0.7, label='Identical Sources') #with poisson
        
        plt.xlabel('Energy (GeV)')
        plt.ylabel('Number of Neutrinos per Energy Bin')
        plt.title('Number of Neutrinos per Energy Bin for Different Cases')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    simulator = BackgroundNeutrinos()
    n_sources = 5000
    events = simulator.simulate_events(n_sources)
    
    # Plot histograms
    simulator.plot_histograms(events)
    
    # Plot skymap
    simulator.plot_skymap(events)

lebins = np.log10(np.linspace(3, 6, 6))
signal_random = SignalNeutrinos(smax=50, nsources=500, verbose=True)
signal_random.createSample(lebins, invisible_fraction=0.1, identity_sources=False, use_poisson=True)

#Plots
signal_random.plot_ra_dec_distribution()
signal_random.plot_skymap()
signal_random.plot_total_flux()
signal_random.plot_flux_per_energy(lebins)
signal_random.plot_neutrinos_per_energy(lebins)