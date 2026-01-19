import xarray as xr
import numpy as np

# This code is written by Adam Massmann and placed in the public
# domain. All warranties are disclaimed.

DAT = '/home/adam/thurston-swb/dat'

def load_ecmwf(file):
    return xr.open_dataset('%s/era5-land-grib-squaxin/%s' % (DAT, file),
                           decode_timedelta=True)

def skookum_location(dataset):
    return dataset.sel(latitude=47.118056, longitude=-123.131388889,
                       method='nearest')

def monthly_average(month, dataset):
    times = range(month-1, len(dataset.time), 12)
    return dataset.isel(time=times).mean()

ecmwf = skookum_location(load_ecmwf('output.nc'))

averages = {
    'Jan' : monthly_average(1, ecmwf),
    'Feb' : monthly_average(2, ecmwf),
    'Mar' : monthly_average(3, ecmwf),
    'Apr' : monthly_average(4, ecmwf),
    'May' : monthly_average(5, ecmwf),
    'Jun' : monthly_average(6, ecmwf),
    'Jul' : monthly_average(7, ecmwf),
    'Aug' : monthly_average(8, ecmwf),
    'Sep' : monthly_average(9, ecmwf),
    'Oct' : monthly_average(10, ecmwf),
    'Nov' : monthly_average(11, ecmwf),
    'Dec' : monthly_average(12, ecmwf)
}

DAYS = {
    'Jan' : 31.0,
    'Feb' : 28.0,
    'Mar' : 31.0,
    'Apr' : 30.0,
    'May' : 31.0,
    'Jun' : 30.0,
    'Jul' : 31.0,
    'Aug' : 31.0,
    'Sep' : 30.0,
    'Oct' : 31.0,
    'Nov' : 30.0,
    'Dec' : 31.0
}

CENTRALIA_EVAP = {
    'Jan' : 0.5,
    'Feb' : 0.7,
    'Mar' : 1.3,
    'Apr' : 2.2,
    'May' : 3.3,
    'Jun' : 3.6,
    'Jul' : 4.7,
    'Aug' : 3.7,
    'Sep' : 2.4,
    'Oct' : 1.2,
    'Nov' : 0.5,
    'Dec' : 0.4
}

def centralia_evap(month):
    """mm/month, from
    Ecology_documents/S2-19990/Bechtel_reservoir_Pages from SWRO100002AF0.pdf"""
    return CENTRALIA_EVAP[month] * 25.4

ALBEDO = 0.07 # middle of small zenith angle, from Arya 2001, see pg 84 of Margulis
VEG_EMISS = 0.98 # pg XX of Margulis
SECONDS_IN_DAY = np.double(60.0*60.0*24.0)

def flux_to_delta_T_per_hour(flux, depth):
    "Flux in W_m2"
    c_p = np.double(4184.0)
    rho = 1000.0
    return flux * 60.0 * 60.0 / (c_p * rho * depth)

WATER_EMISS = 0.97 # pg 19 qual2kw

def shaded_flux_change(averaged):
    "Positive is cooling"
    unshaded_solar = (1.0 - ALBEDO) * averaged['ssrd'].to_numpy() / SECONDS_IN_DAY
    unshaded_longwave = WATER_EMISS * averaged['strd'].to_numpy() / SECONDS_IN_DAY
    shaded_solar = 0.0 # in reality there will be some diffuse radiation!
    canopy_temperature = averaged['t2m'].to_numpy() # could also use skin temperature to span uncertainty
    shaded_longwave = WATER_EMISS * 5.67e-8 * VEG_EMISS * canopy_temperature ** 4.0
    return (unshaded_solar + unshaded_longwave) - (shaded_solar + shaded_longwave)

# Esimate velocity w/ manning roughness
# about 700m based on google maps
LENGTH = 700.0 # m, based on google maps
def manning(radius, slope, manning_n):
    return radius**(2.0 / 3.0) * np.sqrt(slope) / manning_n

def hydraulic_radius(width, depth):
    "assume rectangular"
    area = width * depth
    perimeter = 2 * depth + width
    return area / perimeter

WIDTH = 5.0 # m
UPSTREAM_ELEVATION = 14.6 # based on google earth
DOWNSTREAM_ELEVATION = 14.0 # based on google earth
ELEVATION_DROP = UPSTREAM_ELEVATION-DOWNSTREAM_ELEVATION
ROUGHNESS = 0.05 # "weeds and pools, winding", pg 16-17 of qual2kw

def residence_time_hr(length, width, depth, ELEVATION_DROP, roughness):
    slope = ELEVATION_DROP/LENGTH
    velocity = manning(hydraulic_radius(width, depth), slope, roughness)
    time = length / velocity
    return time / (60.0 * 60.0)

DEPTH = 0.3 # m
RESIDENCE_TIME = residence_time_hr(LENGTH, WIDTH, DEPTH, ELEVATION_DROP, ROUGHNESS)
print('Residence time: %.5f hr' %
      (RESIDENCE_TIME))

print('Month & Cooling Flux (W/m$^{2}$) & Rate Temperature Change (C/hr) & Approx Total Change (C) \\\\')
for month  in averages.keys():
    average = averages[month]
    flux = np.average(shaded_flux_change(average)) # average over uncertainty
    rate = flux_to_delta_T_per_hour(flux, DEPTH)
    print('%s & %.2f & %.2f & %.2f \\\\' % (month, flux, rate, RESIDENCE_TIME * rate))

def load_invariant(var_string):
    fname = '%s_0.area-subset.47.5.-122.46.4.-123.6.nc' % var_string
    dataset = skookum_location(load_ecmwf(fname))
    if var_string == 'clake':
        var_string = 'cl'
    return dataset[var_string].to_numpy()

veg_high_cover = load_invariant('cvh')
veg_high_type = load_invariant('tvh') # https://codes.ecmwf.int/grib/format/grib2/ctables/4/234/
veg_low_cover = load_invariant('cvl')
veg_low_type = load_invariant('tvl')
lake_cover = load_invariant('clake')

def m_per_day_to_mm_per_month(m, month):
    n_days = DAYS[month]
    return m * 1000.0 * n_days

print('Month & Potential Evaporation (mm) & Centralia Reservior Evap (mm) & Forest Evapotranspiration (mm) \\\\')
for month  in averages.keys():
    average = averages[month]
    pe = m_per_day_to_mm_per_month(-1.0*average['pev'], month)
    e = m_per_day_to_mm_per_month(-1.0*average['e'], month)
    print('%s & %.2f & %.2f & %.2f \\\\'
          % (month, pe, centralia_evap(month), e))


