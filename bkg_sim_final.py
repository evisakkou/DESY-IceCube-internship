import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

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

        fig.savefig('Histograms.png')

        # Combined histogram for energies
        plt.figure(figsize=(10, 5))
        plt.hist([np.log10(events['atm bkg'][2]), np.log10(events['astr bkg'][2])], bins=60, color=['blue', 'red'], alpha=0.5, label=['Atm Bkg', 'Astr Bkg'])
        plt.xlabel('log10(Energy)')
        # plt.ylabel('Number of Sources')
        plt.title('Combined Histogram of Energy')
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
            hp.mollview(skymap, title=f"Skymap of {event_type.capitalize()} Sources", unit="Number of Sources", cmap="viridis")
            hp.graticule()
            plt.show()

if __name__ == "__main__":
    simulator = BackgroundNeutrinos()
    n_sources = 5000
    events = simulator.simulate_events(n_sources)
    
    # Plot histograms
    simulator.plot_histograms(events)
    
    # Plot skymap
    simulator.plot_skymap(events)
