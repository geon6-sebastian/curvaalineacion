import math
import numpy as np
import numba
import sys
from dataclasses import dataclass

DEBUG = False

#elipsoide GRS80
GRS80_a = 6378137.0
GRS80_f = 1.0/298.2572221008827

EPSILON = 1E-15
PI = math.pi
PI_2 = 0.5 * PI
PI_4 = 0.5 * PI_2

MIN_FLOAT = sys.float_info.min
MAX_FLOAT = sys.float_info.max

SQRT_MAX_FLOAT = math.sqrt(MAX_FLOAT)
SQRT_MIN_FLOAT = math.sqrt(MIN_FLOAT)

def tosemicirc(num : float) :
	result = math.remainder(num, 2.0 * PI)
	if result == -PI:
		result = PI
	return result

@numba.njit
def sincos(x):
    return np.sin(x), np.cos(x)

@numba.njit
def arccos(x):
	x_clipped = np.clip(x, -1.0, 1.0)
	# arccos(x) = arctan2(√(1-x²), x)
	#return np.arctan2(np.sqrt(1.0 - x_clipped**2), x_clipped)
	return arg(x_clipped, np.sqrt(1.0 - x_clipped*x_clipped))

@numba.njit
def arcsen(x):
	x_clipped = np.clip(x, -1.0, 1.0)
	return arg(np.sqrt(1.0 - x_clipped*x_clipped), x_clipped)

@numba.njit
def sgn(num : float) :
    if (num < 0.0) :
        return -1.0
    else :
        return 1.0

@numba.njit
def DIV(y:float , x:float):
	sx = sgn(x)
	return y/(x + sx*MIN_FLOAT*2.0)

@numba.njit
def RAD2DEG(alpha : float) :
    	return alpha*180.0/math.pi

@numba.njit
def DEG2RAD(alpha : float) :
   	return alpha/180.0*math.pi

@numba.njit
def SQR(x : float) :
	if math.fabs(x) < SQRT_MAX_FLOAT :
		if math.fabs(x) > SQRT_MIN_FLOAT :
			return x*x
		else :
			return MIN_FLOAT
	else :
		return MAX_FLOAT

@numba.njit
def arg(x : float , y : float) :
	if (math.fabs(y) > EPSILON ) :
		rho = math.hypot(x,y)
		return 2.0*math.atan(DIV(y , rho + x))
	else :
		if (x >= 0.0) :
			return 0.0
		else :
			return math.pi

#-------------------------------------------------------------------------------------------
@dataclass
class pelipsoide_t :
	a : float = 0.0
	f : float = 0.0
	b : float = 0.0
	c : float = 0.0
	n : float = 0.0
	n2 : float = 0.0
	n3 : float = 0.0
	n4 : float = 0.0
	n5 : float = 0.0
	n6 : float = 0.0
	e2 : float = 0.0
	e : float = 0.0
	e4 : float = 0.0
	e6 : float = 0.0
	e8 : float = 0.0
	e10 : float = 0.0
	ep2 : float = 0.0
	ep : float = 0.0
	R1 : float = 0.0
	R2 : float = 0.0
	R3 : float = 0.0
	Rm : float = 0.0
	Rmu : float = 0.0
	Q : float = 0.0
	K : float = 0.0

def pelipsoide_init(in_f: float, in_a: float) :
	p = pelipsoide_t()
	p.a = in_a
	p.f = in_f
	p.b = p.a*(1.0 - p.f)
	p.c = p.a/(1.0 - p.f)
	p.n = p.f/(2.0 - p.f)
	p.n2 = p.n*p.n
	p.n3 = p.n2*p.n
	p.n4 = p.n2*p.n2
	p.n5 = p.n4*p.n
	p.n6 = p.n5*p.n
	p.e2 = p.f*(2.0 - p.f)
	p.e = math.sqrt(p.e2)
	p.e4 = p.e2*p.e2
	p.e6 = p.e4*p.e2
	p.e8 = p.e4*p.e4
	p.e10 = p.e6*p.e4
	p.ep2 = p.e2/(1.0 - p.e2)
	p.ep = math.sqrt(p.ep2)
	p.R1 = (2.0*p.a + p.b)/3.0
	p.R2 = p.a*math.sqrt(0.5+0.5*(DIV(1.0,p.e) - p.e)* math.atanh(p.e)) #//radiio autálico
	p.R3 = np.cbrt(p.a*p.a*p.b)
	p.Rm = p.a/(1.0 + p.n)
	p.Rmu = p.Rm*(1.0 +p.n2/4.0 +p.n4/64.0 +p.n6/256.0)
	return p

@dataclass
class platn6_t:
	GEOD2RECT = [0.0 for i in range(7)] 
	RECT2GEOD = [0.0 for i in range(7)]
	ISOM2GEOD = [0.0 for i in range(7)]
	GEOD2CONF = [0.0 for i in range(7)]
	CONF2GEOD = [0.0 for i in range(7)]
	GEOD2AUTH = [0.0 for i in range(7)]
	AUTH2GEOD = [0.0 for i in range(7)]
	GEOD2GEOC = [0.0 for i in range(7)]
	GEOC2GEOD = [0.0 for i in range(7)]
	GEOD2PARA = [0.0 for i in range(7)]
	PARA2GEOD = [0.0 for i in range(7)]
	CONF2RECT = [0.0 for i in range(7)]
	AUTH2RECT = [0.0 for i in range(7)]
	RECT2CONF = [0.0 for i in range(7)]
	RECT2AUTH = [0.0 for i in range(7)]
	GEOC2RECT = [0.0 for i in range(7)]
	RECT2GEOC = [0.0 for i in range(7)]
	PARA2RECT = [0.0 for i in range(7)]
	RECT2PARA = [0.0 for i in range(7)]
	AUTH2CONF = [0.0 for i in range(7)]
	CONF2AUTH = [0.0 for i in range(7)]
	GEOC2CONF = [0.0 for i in range(7)]
	CONF2GEOC = [0.0 for i in range(7)]
	PARA2CONF = [0.0 for i in range(7)]
	CONF2PARA = [0.0 for i in range(7)]
	GEOC2AUTH = [0.0 for i in range(7)]
	AUTH2GEOC = [0.0 for i in range(7)]
	PARA2AUTH = [0.0 for i in range(7)]
	AUTH2PARA = [0.0 for i in range(7)]
	PARA2GEOC = [0.0 for i in range(7)]
	GEOC2PARA = [0.0 for i in range(7)]

def init_platn6(n : float):
	n2 = n*n
	n3 = n2*n
	n4 = n2*n2
	n5 = n3*n2
	n6 = n3*n3
	p = platn6_t()
	p.GEOD2RECT[0] = 0.0
	p.GEOD2RECT[1] = -3.0/2.0*n +9.0/16.0*n3 -3.0/32.0*n5
	p.GEOD2RECT[2] = 15.0/16.0*n2 -15.0/32.0*n4 +135.0/2048.0*n6
	p.GEOD2RECT[3]= -35.0/48.0*n3 +105.0/256.0*n5
	p.GEOD2RECT[4] = 315.0/512.0*n4 -189.0/512.0*n6
	p.GEOD2RECT[5] = -693.0/1280.0*n5
	p.GEOD2RECT[6] = 1001.0/2048.0*n6
	p.RECT2GEOD[0] = 0.0
	p.RECT2GEOD[1] = 3.0/2.0*n -27.0/32.0*n3 +269.0/512.0*n5
	p.RECT2GEOD[2] = 21.0/16.0*n2 -55.0/32.0*n4 +6759.0/4096.0*n6
	p.RECT2GEOD[3] = 151.0/96.0*n3 -417.0/128.0*n5
	p.RECT2GEOD[4] = 1097.0/512.0*n4 -15543.0/2560.0*n6
	p.RECT2GEOD[5] = 8011.0/2560.0*n5
	p.RECT2GEOD[6] = 293393.0/61440.0*n6

	p.ISOM2GEOD[0] = 0.0
	p.ISOM2GEOD[1] = 4.0*n -32.0/3.0*n2 +124.0/5.0*n3 -3296.0/63.0*n4 +32476.0/315.0*n5 -30081056.0/155925.0*n6
	p.ISOM2GEOD[2] = 56.0/3.0*n2 -1984.0/15.0*n3 +65872.0/105.0*n4 -764096.0/315.0*n5 +85344488.0/10395.0*n6
	p.ISOM2GEOD[3] = 1792.0/15.0*n3 -149984.0/105.0*n4 +1085824.0/105.0*n5 -1801378688.0/31185.0*n6
	p.ISOM2GEOD[4] = 273856.0/315.0*n4 -931328.0/63.0*n5 +4510583296.0/31185.0*n6
	p.ISOM2GEOD[5] = 2137088.0/315.0*n5 -57822208.0/385.0*n6
	p.ISOM2GEOD[6] = 1232232448.0/22275.0*n6

	p.GEOD2CONF[0] = 0.0
	p.GEOD2CONF[1] = -2.0*n +2.0/3.0*n2 +4.0/3.0*n3 -82.0/45.0*n4 +32.0/45.0*n5 +4642.0/4725.0*n6
	p.GEOD2CONF[2] = 5.0/3.0*n2 -16.0/15.0*n3 -13.0/9.0*n4 +904.0/315.0*n5 -1522.0/945.0*n6
	p.GEOD2CONF[3] = -26.0/15.0*n3 +34.0/21.0*n4 +8.0/5.0*n5 -12686.0/2835.0*n6
	p.GEOD2CONF[4] = 1237.0/630.0*n4 -12.0/5.0*n5 -24832.0/14175.0*n6
	p.GEOD2CONF[5] = -734.0/315.0*n5 +109598.0/31185.0*n6
	p.GEOD2CONF[6] = 444337.0/155925.0*n6
	p.CONF2GEOD[0] = 0.0
	p.CONF2GEOD[1] = 2.0*n -2.0/3.0*n2 -2*n3 +116/45*n4 +26/45*n5 -2854/675*n6
	p.CONF2GEOD[2] = 7.0/3.0*n2 -8.0/5.0*n3 -227.0/45.0*n4 +2704.0/315.0*n5 +2323.0/945.0*n6
	p.CONF2GEOD[3] = 56.0/15.0*n3 -136.0/35.0*n4 -1262.0/105.0*n5 +73814.0/2835.0*n6
	p.CONF2GEOD[4] = 4279.0/630.0*n4 -332.0/35.0*n5 -399572.0/14175.0*n6
	p.CONF2GEOD[5] = 4174.0/315.0*n5 -144838.0/6237.0*n6
	p.CONF2GEOD[6] = 601676.0/22275.0*n6

	p.GEOD2AUTH[0] = 0.0
	p.GEOD2AUTH[1] = -4.0/3.0*n -4.0/45.0*n2 +88.0/315.0*n3 +538.0/4725.0*n4 +20824.0/467775.0*n5 -44732.0/2837835.0*n6
	p.GEOD2AUTH[2] = 34.0/45.0*n2 +8.0/105.0*n3 -2482.0/14175.0*n4 -37192.0/467775.0*n5 -12467764.0/212837625.0*n6
	p.GEOD2AUTH[3] = -1532.0/2835.0*n3 -898.0/14175.0*n4 +54968.0/467775.0*n5 +100320856.0/1915538625.0*n6
	p.GEOD2AUTH[4] = 6007.0/14175.0*n4 +24496.0/467775.0*n5 -5884124.0/70945875.0*n6
	p.GEOD2AUTH[5] = -23356.0/66825.0*n5 -839792.0/19348875.0*n6
	p.GEOD2AUTH[6] = 570284222.0/1915538625.0*n6
	p.AUTH2GEOD[0] = 0.0
	p.AUTH2GEOD[1] = 4.0/3.0*n +4.0/45.0*n2 -16.0/35.0*n3 -2582.0/14175.0*n4 +60136.0/467775.0*n5 +28112932.0/212837625.0*n6
	p.AUTH2GEOD[2] = 46.0/45.0*n2 +152.0/945.0*n3 -11966.0/14175.0*n4 -21016.0/51975.0*n5 +251310128.0/638512875.0*n6
	p.AUTH2GEOD[3] = 3044.0/2835.0*n3 +3802.0/14175.0*n4 -94388.0/66825.0*n5 -8797648.0/10945935.0*n6
	p.AUTH2GEOD[4] = 6059.0/4725.0*n4 +41072.0/93555.0*n5 -1472637812.0/638512875.0*n6
	p.AUTH2GEOD[5] = 768272.0/467775.0*n5 +455935736.0/638512875.0*n6
	p.AUTH2GEOD[6] = 4210684958.0/1915538625.0*n6

	p.GEOD2GEOC[0] = 0.0
	p.GEOD2GEOC[1] = -2.0*n +2.0*n3 -2.0*n5
	p.GEOD2GEOC[2] = 2.0*n2 -4.0*n4 +6.0*n6
	p.GEOD2GEOC[3] = -8.0/3.0*n3 +8.0*n5
	p.GEOD2GEOC[4] = 4.0*n4 -16.0*n6
	p.GEOD2GEOC[5] = -32.0/5.0*n5
	p.GEOD2GEOC[6] = 32.0/3.0*n6
	p.GEOC2GEOD[0] = 0.0
	p.GEOC2GEOD[1] = 2.0*n -2.0*n3 +2.0*n5
	p.GEOC2GEOD[2] = 2.0*n2 -4.0*n4 +6.0*n6
	p.GEOC2GEOD[3] = 8.0/3.0*n3 -8.0*n5
	p.GEOC2GEOD[4] = 4.0*n4 -16.0*n6
	p.GEOC2GEOD[5] = 32.0/5.0*n5
	p.GEOC2GEOD[6] = 32.0/3.0*n6

	p.GEOD2PARA[0] = 0.0
	p.GEOD2PARA[1] = -n
	p.GEOD2PARA[2] = 1.0/2.0*n2
	p.GEOD2PARA[3] = -1.0/3.0*n3
	p.GEOD2PARA[4] = 1.0/4.0*n4
	p.GEOD2PARA[5] = -1.0/5.0*n5
	p.GEOD2PARA[6] = 1.0/6.0*n6
	p.PARA2GEOD[0] = 0
	p.PARA2GEOD[1] = n
	p.PARA2GEOD[2] = 1.0/2.0*n2
	p.PARA2GEOD[3] = 1.0/3.0*n3
	p.PARA2GEOD[4] = 1.0/4.0*n4
	p.PARA2GEOD[5] = 1.0/5.0*n5
	p.PARA2GEOD[6] = 1.0/6.0*n6

	p.CONF2RECT[0] = 0.0
	p.CONF2RECT[1] = 1.0/2.0*n -2.0/3.0*n2 +5.0/16.0*n3 +41.0/180.0*n4 -127.0/288.0*n5 +7891.0/37800.0*n6
	p.CONF2RECT[2] = 13.0/48.0*n2 -3.0/5.0*n3 +557.0/1440.0*n4 +281.0/630.0*n5 -1983433.0/1935360.0*n6
	p.CONF2RECT[3] = 61.0/240.0*n3 -103.0/140.0*n4 +15061.0/26880.0*n5 +167603.0/181440.0*n6
	p.CONF2RECT[4] = 49561.0/161280.0*n4 -179.0/168.0*n5 +6601661.0/7257600.0*n6
	p.CONF2RECT[5] = 34729.0/80640.0*n5 -3418889.0/1995840.0*n6
	p.CONF2RECT[6] = 212378941.0/319334400.0*n6
	p.RECT2CONF[0] = 0.0
	p.RECT2CONF[1] = -1.0/2.0*n +2.0/3.0*n2 -37.0/96.0*n3 +1.0/360.0*n4 +81.0/512.0*n5 -96199.0/604800.0*n6
	p.RECT2CONF[2] = -1.0/48.0*n2 -1.0/15.0*n3 +437.0/1440.0*n4 -46.0/105.0*n5 +1118711.0/3870720.0*n6
	p.RECT2CONF[3] = -17.0/480.0*n3 +37.0/840.0*n4 +209.0/4480.0*n5 -5569.0/90720.0*n6
	p.RECT2CONF[4] = -4397.0/161280.0*n4 +11.0/504.0*n5 +830251.0/7257600.0*n6
	p.RECT2CONF[5] = -4583.0/161280.0*n5 +108847.0/3991680.0*n6
	p.RECT2CONF[6] = -20648693.0/638668800.0*n6

	p.AUTH2RECT[0] = 0.0
	p.AUTH2RECT[1] = -1.0/6.0*n +4.0/45.0*n2 +121.0/1680.0*n3 -1609.0/28350.0*n4 -384229.0/14968800.0*n5 +12674323.0/851350500.0*n6
	p.AUTH2RECT[2] = -29.0/720.0*n2 +26.0/945.0*n3 +16463.0/453600.0*n4 -431.0/17325.0*n5 -31621753811.0/1307674368000.0*n6
	p.AUTH2RECT[3] = -1003.0/45360.0*n3 +449.0/28350.0*n4 +3746047.0/119750400.0*n5 -32844781.0/1751349600.0*n6
	p.AUTH2RECT[4] = -40457.0/2419200.0*n4 +629.0/53460.0*n5 +10650637121.0/326918592000.0*n6
	p.AUTH2RECT[5] = -1800439.0/119750400.0*n5 +205072597.0/20432412000.0*n6
	p.AUTH2RECT[6] = -59109051671.0/3923023104000.0*n6
	p.RECT2AUTH[0] = 0.0
	p.RECT2AUTH[1] = 1.0/6.0*n -4.0/45.0*n2 -817.0/10080.0*n3 +1297.0/18900.0*n4 +7764059.0/239500800.0*n5 -9292991.0/302702400.0*n6
	p.RECT2AUTH[2] = 49.0/720.0*n2 -2.0/35.0*n3 -29609.0/453600.0*n4 +35474.0/467775.0*n5 +36019108271.0/871782912000.0*n6
	p.RECT2AUTH[3] = 4463.0/90720.0*n3 -2917.0/56700.0*n4 -4306823.0/59875200.0*n5 +3026004511.0/30648618000.0*n6
	p.RECT2AUTH[4] = 331799.0/7257600.0*n4 -102293.0/1871100.0*n5 -368661577.0/4036032000.0*n6
	p.RECT2AUTH[5] = 11744233.0/239500800.0*n5 -875457073.0/13621608000.0*n6
	p.RECT2AUTH[6] = 453002260127.0/7846046208000.0*n6

	p.GEOC2RECT[0] = 0.0
	p.GEOC2RECT[1] = 1.0/2.0*n +13.0/16.0*n3 -15.0/32.0*n5
	p.GEOC2RECT[2] = -1.0/16.0*n2 +33.0/32.0*n4 -1673.0/2048.0*n6
	p.GEOC2RECT[3] = -5.0/16.0*n3 +349.0/256.0*n5
	p.GEOC2RECT[4] = -261.0/512.0*n4 +963.0/512.0*n6
	p.GEOC2RECT[5] = -921.0/1280.0*n5
	p.GEOC2RECT[6] = -6037.0/6144.0*n6
	p.RECT2GEOC[0] = 0.0
	p.RECT2GEOC[1] = -1.0/2.0*n -23.0/32.0*n3 +499.0/1536.0*n5
	p.RECT2GEOC[2] = 5.0/16.0*n2 -5.0/96.0*n4 +6565.0/12288.0*n6
	p.RECT2GEOC[3] = 1.0/32.0*n3 -77.0/128.0*n5
	p.RECT2GEOC[4] = 283.0/1536.0*n4 -4037.0/7680.0*n6
	p.RECT2GEOC[5] = 1301.0/7680.0*n5
	p.RECT2GEOC[6] = 17089.0/61440.0*n6

	p.PARA2RECT[0] = 0.0
	p.PARA2RECT[1] = -1.0/2.0*n +3.0/16.0*n3 -1.0/32.0*n5
	p.PARA2RECT[2] = -1.0/16.0*n2 +1.0/32.0*n4 -9.0/2048.0*n6
	p.PARA2RECT[3] = -1.0/48.0*n3 +3.0/256.0*n5
	p.PARA2RECT[4] = -5.0/512.0*n4 +3.0/512.0*n6
	p.PARA2RECT[5] = -7.0/1280.0*n5
	p.PARA2RECT[6] = -7.0/2048.0*n6
	p.RECT2PARA[0] = 0.0
	p.RECT2PARA[1] = 1.0/2.0*n -9.0/32.0*n3 +205.0/1536.0*n5
	p.RECT2PARA[2] = 5.0/16.0*n2 -37.0/96.0*n4 +1335.0/4096.0*n6
	p.RECT2PARA[3] = 29.0/96.0*n3 -75.0/128.0*n5
	p.RECT2PARA[4] = 539.0/1536.0*n4 -2391.0/2560.0*n6
	p.RECT2PARA[5] = 3467.0/7680.0*n5
	p.RECT2PARA[6] = 38081.0/61440.0*n6

	p.AUTH2CONF[0] = 0.0
	p.AUTH2CONF[1] = -2.0/3.0*n +34.0/45.0*n2 -88.0/315.0*n3 -2312.0/14175.0*n4 +27128.0/93555.0*n5 -55271278.0/212837625.0*n6
	p.AUTH2CONF[2] = 1.0/45.0*n2 -184.0/945.0*n3 +6079.0/14175.0*n4 -65864.0/155925.0*n5 +106691108.0/638512875.0*n6
	p.AUTH2CONF[3] = -106.0/2835.0*n3 +772.0/14175.0*n4 -14246.0/467775.0*n5 +5921152.0/54729675.0*n6
	p.AUTH2CONF[4] = -167.0/9450.0*n4 -5312.0/467775.0*n5 +75594328.0/638512875.0*n6
	p.AUTH2CONF[5] = -248.0/13365.0*n5 +2837636.0/638512875.0*n6
	p.AUTH2CONF[6] = -34761247.0/1915538625.0*n6
	p.CONF2AUTH[0] = 0.0
	p.CONF2AUTH[1] = 2.0/3.0*n -34.0/45.0*n2 +46.0/315.0*n3 +2458.0/4725.0*n4 -55222.0/93555.0*n5 +2706758.0/42567525.0*n6
	p.CONF2AUTH[2] = 19.0/45.0*n2 -256.0/315.0*n3 +3413.0/14175.0*n4 +516944.0/467775.0*n5 -340492279.0/212837625.0*n6
	p.CONF2AUTH[3] = 248.0/567.0*n3 -15958.0/14175.0*n4 +206834.0/467775.0*n5 +4430783356.0/1915538625.0*n6
	p.CONF2AUTH[4] = 16049.0/28350.0*n4 -832976.0/467775.0*n5 +62016436.0/70945875.0*n6
	p.CONF2AUTH[5] = 15602.0/18711.0*n5 -651151712.0/212837625.0*n6
	p.CONF2AUTH[6] = 2561772812.0/1915538625.0*n6

	p.GEOC2CONF[0] = 0.0
	p.GEOC2CONF[1] = 2.0/3.0*n2 +2.0/3.0*n3 -2.0/9.0*n4 -14.0/45.0*n5 +1042.0/4725.0*n6
	p.GEOC2CONF[2] = -1.0/3.0*n2 +4.0/15.0*n3 +43.0/45.0*n4 -4.0/45.0*n5 -712.0/945.0*n6
	p.GEOC2CONF[3] = -2.0/5.0*n3 +2.0/105.0*n4 +124.0/105.0*n5 +274.0/2835.0*n6
	p.GEOC2CONF[4] = -55.0/126.0*n4 -16.0/105.0*n5 +21068.0/14175.0*n6
	p.GEOC2CONF[5] = -22.0/45.0*n5 -9202.0/31185.0*n6
	p.GEOC2CONF[6] = -90263.0/155925.0*n6
	p.CONF2GEOC[0] = 0.0
	p.CONF2GEOC[1] = -2.0/3.0*n2 -2.0/3.0*n3 +4.0/9.0*n4 +2.0/9.0*n5 -3658.0/4725.0*n6
	p.CONF2GEOC[2] = 1.0/3.0*n2 -4.0/15.0*n3 -23.0/45.0*n4 +68.0/45.0*n5 +61.0/135.0*n6
	p.CONF2GEOC[3] = 2.0/5.0*n3 -24.0/35.0*n4 -46.0/35.0*n5 +9446.0/2835.0*n6
	p.CONF2GEOC[4] = 83.0/126.0*n4 -80.0/63.0*n5 -34712.0/14175.0*n6
	p.CONF2GEOC[5] = 52.0/45.0*n5 -2362.0/891.0*n6
	p.CONF2GEOC[6] = 335882.0/155925.0*n6

	p.PARA2CONF[0] = 0.0
	p.PARA2CONF[1] = -n +2.0/3.0*n2 -16.0/45.0*n4 +2.0/5.0*n5 -998.0/4725.0*n6
	p.PARA2CONF[2] = 1.0/6.0*n2 -2.0/5.0*n3 +19.0/45.0*n4 -22.0/105.0*n5 -2.0/27.0*n6
	p.PARA2CONF[3] = -1.0/15.0*n3 +16.0/105.0*n4 -22.0/105.0*n5 +116.0/567.0*n6
	p.PARA2CONF[4] = 17.0/1260.0*n4 -8.0/105.0*n5 +2123.0/14175.0*n6
	p.PARA2CONF[5] = -1.0/105.0*n5 +128.0/4455.0*n6
	p.PARA2CONF[6] = 149.0/311850.0*n6
	p.CONF2PARA[0] = 0.0
	p.CONF2PARA[1] = n -2.0/3.0*n2 -1.0/3.0*n3 +38.0/45.0*n4 -1.0/3.0*n5 -3118.0/4725.0*n6
	p.CONF2PARA[2] = 5.0/6.0*n2 -14.0/15.0*n3 -7.0/9.0*n4 +50.0/21.0*n5 -247.0/270.0*n6
	p.CONF2PARA[3] = 16.0/15.0*n3 -34.0/21.0*n4 -5.0/3.0*n5 +17564.0/2835.0*n6
	p.CONF2PARA[4] = 2069.0/1260.0*n4 -28.0/9.0*n5 -49877.0/14175.0*n6
	p.CONF2PARA[5] = 883.0/315.0*n5 -28244.0/4455.0*n6
	p.CONF2PARA[6] = 797222.0/155925.0*n6

	p.GEOC2AUTH[0] = 0.0
	p.GEOC2AUTH[1] = 2.0/3.0*n -4.0/45.0*n2 +62.0/105.0*n3 +778.0/4725.0*n4 -193082.0/467775.0*n5 -4286228.0/42567525.0*n6
	p.GEOC2AUTH[2] = 4.0/45.0*n2 -32.0/315.0*n3 +12338.0/14175.0*n4 +92696.0/467775.0*n5 -61623938.0/70945875.0*n6
	p.GEOC2AUTH[3] = -524.0/2835.0*n3 -1618.0/14175.0*n4 +612536.0/467775.0*n5 +427003576.0/1915538625.0*n6
	p.GEOC2AUTH[4] = -5933.0/14175.0*n4 -8324.0/66825.0*n5 +427770788.0/212837625.0*n6
	p.GEOC2AUTH[5] = -320044.0/467775.0*n5 -9153184.0/70945875.0*n6
	p.GEOC2AUTH[6] = -1978771378.0/1915538625.0*n6
	p.AUTH2GEOC[0] = 0.0
	p.AUTH2GEOC[1] = -2.0/3.0*n +4.0/45.0*n2 -158.0/315.0*n3 -2102.0/14175.0*n4 +109042.0/467775.0*n5 +216932.0/2627625.0*n6
	p.AUTH2GEOC[2] = 16.0/45.0*n2 -16.0/945.0*n3 +934.0/14175.0*n4 -7256.0/155925.0*n5 +117952358.0/638512875.0*n6
	p.AUTH2GEOC[3] = -232.0/2835.0*n3 +922.0/14175.0*n4 -25286.0/66825.0*n5 -7391576.0/54729675.0*n6
	p.AUTH2GEOC[4] = 719.0/4725.0*n4 +268.0/18711.0*n5 -67048172.0/638512875.0*n6
	p.AUTH2GEOC[5] = 14354.0/467775.0*n5 +46774256.0/638512875.0*n6
	p.AUTH2GEOC[6] = 253129538.0/1915538625.0*n6

	p.PARA2AUTH[0] = 0.0
	p.PARA2AUTH[1] = -1.0/3.0*n -4.0/45.0*n2 +32.0/315.0*n3 +34.0/675.0*n4 +2476.0/467775.0*n5 -70496.0/8513505.0*n6
	p.PARA2AUTH[2] = -7.0/90.0*n2 -4.0/315.0*n3 +74.0/2025.0*n4 +3992.0/467775.0*n5 +53836.0/212837625.0*n6
	p.PARA2AUTH[3] = -83.0/2835.0*n3 +2.0/14175.0*n4 +7052.0/467775.0*n5 -661844.0/1915538625.0*n6
	p.PARA2AUTH[4] = -797.0/56700.0*n4 +934.0/467775.0*n5 +1425778.0/212837625.0*n6
	p.PARA2AUTH[5] = -3673.0/467775.0*n5 +390088.0/212837625.0*n6
	p.PARA2AUTH[6] = -18623681.0/3831077250.0*n6
	p.AUTH2PARA[0] = 0.0
	p.AUTH2PARA[1] = 1.0/3.0*n +4.0/45.0*n2 -46.0/315.0*n3 -1082.0/14175.0*n4 +11824.0/467775.0*n5 +7947332.0/212837625.0*n6
	p.AUTH2PARA[2] = 17.0/90.0*n2 +68.0/945.0*n3 -338.0/2025.0*n4 -16672.0/155925.0*n5 +39946703.0/638512875.0*n6
	p.AUTH2PARA[3] = 461.0/2835.0*n3 +1102.0/14175.0*n4 -101069.0/467775.0*n5 -255454.0/1563705.0*n6
	p.AUTH2PARA[4] = 3161.0/18900.0*n4 +1786.0/18711.0*n5 -189032762.0/638512875.0*n6
	p.AUTH2PARA[5] = 88868.0/467775.0*n5 +80274086.0/638512875.0*n6
	p.AUTH2PARA[6] = 880980241.0/3831077250.0*n6

	p.PARA2GEOC[0] = 0.0
	p.PARA2GEOC[1] = -n
	p.PARA2GEOC[2] =1.0/2.0*n2
	p.PARA2GEOC[3] = -1.0/3.0*n3
	p.PARA2GEOC[4] = 1.0/4.0*n4
	p.PARA2GEOC[5] = -1.0/5.0*n5
	p.PARA2GEOC[6] = 1.0/6.0*n6
	p.GEOC2PARA[0] = 0.0
	p.GEOC2PARA[1] = n
	p.GEOC2PARA[2] = 1.0/2.0*n2
	p.GEOC2PARA[3] = 1.0/3.0*n3
	p.GEOC2PARA[4] = 1.0/4.0*n4
	p.GEOC2PARA[5] = 1.0/5.0*n5
	p.GEOC2PARA[6] = 1.0/6.0*n6
	return p

def lat2latn6(a, LAT1 : float) :
	#a son los parámetros obtenidos de n
	#esta es la forma de horner
	cp = math.cos(2.0*LAT1)
	LAT2 = LAT1 +(cp*(cp*(cp*(cp*(32.0*a[6]*cp+16.0*a[5])-32.0*a[6]+8.0*a[4])-12.0*a[5]+4.0*a[3])+6.0*a[6]-4.0*a[4]+2.0*a[2])+a[5]-a[3]+a[1])*math.sin(2.0*LAT1)
	return LAT2

def partial_from_latn6(a, phi : float):
	cp = math.cos(phi)
	cp2 = cp*cp
	#float(horner(trigsimp(trigexpand(diff(sum(a[j]*sin(2*j*phi), j, 1, 6),phi)))));
	latp = cp2*(cp2*(cp2*(cp2*(cp2*(24576.0*a[6]*cp2-73728.0*a[6]+5120.0*a[5])+82944.0*a[6]-12800.0*a[5]+1024.0*a[4])
	-43008.0*a[6]+11200.0*a[5]-2048.0*a[4]+192.0*a[3])+10080.0*a[6]-4000.0*a[5]+1280.0*a[4]-288.0*a[3]+32.0*a[2])-864.0*a[6]+500.0*a[5]
	-256.0*a[4]+108.0*a[3]-32.0*a[2]+4.0*a[1])+12.0*a[6]-10.0*a[5]+8.0*a[4]-6.0*a[3]+4.0*a[2]-2.0*a[1]
	return(1.0 + latp)

def radios_ell(c : float , ep : float , latitude : float) :
	cosphi = math.cos(latitude)
	V = math.hypot(1.0 , ep*cosphi)
	RN = c/V
	RM = c/(V*V*V)
	RG = c/(V*V)
	r = RN*cosphi
	return RN , RM , RG , r

def phi2psi(elli, phi):
	u = math.sin(phi)*(1.0 - elli.e2)
	v = math.cos(phi)
	return arg(v, u)

def psi2phi(elli, psi):
	u = math.sin(psi)
	v = math.cos(psi)*(1.0 - elli.e2)
	return arg(v, u)

def psi_partial_phi(elli, phi):
	u = 1.0 + elli.ep2
	v = 1.0 + (2.0 + elli.ep2)*elli.ep2*math.cos(phi)**2
	return u/v

def phi2beta(elli, phi):
	beta = arg(math.cos(phi), (1.0 - elli.f)*math.sin(phi))
	return beta

def beta2phi(elli, beta):
	phi = arg((1.0 - elli.f)*math.cos(beta), math.sin(beta))
	return phi

def beta2tau(elli, beta):
    t = elli.ep*math.sin(beta)
    k = elli.a/elli.R2*elli.b/elli.R2*0.5/elli.ep
    tau = math.asin(k*(math.asinh(t) + t*math.hypot(1,t)))
    return tau

def beta_partial_phi(elli, phi):
	return (1.0 - elli.f)/(1.0 - elli.f*(2.0 - elli.f)*math.sin(phi)**2)

#-------------------------------------------------------------------------------------------
EPSILON_MAQ = 1E-16
EPSILON_ANG = 1E-10
MAXSTEP_ANG = DEG2RAD(5.0)
MAXSTEPRK45_ANG_INITIALVALUE = DEG2RAD(0.1)
MAXSTEP_h = 1E-5
MAXDIST = 10.0

@dataclass
class pRK45_t:
    tol: float
    h_max: float
    h_min: float

# --- Coeficientes del método Dormand-Prince (RK45) ---
A = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
B = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [1/5, 0, 0, 0, 0, 0, 0],
    [3/40, 9/40, 0, 0, 0, 0, 0],
    [44/45, -56/15, 32/9, 0, 0, 0, 0],
    [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
    [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
    [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
])
C5 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
C4 = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])

def cs_align_(elli, beta1:float, L1:float, beta2:float, L2:float):
    a = elli.a; b = elli.b
    cbeta1 = math.cos(beta1); sbeta1 = math.sin(beta1)
    cL1 = math.cos(L1); sL1 = math.sin(L1)
    cbeta2 = math.cos(beta2); sbeta2 = math.sin(beta2)
    cL2 = math.cos(L2); sL2 = math.sin(L2)
    X1 = a*cbeta1*cL1; Y1 = a*cbeta1*sL1; Z1 = b*sbeta1
    X2 = a*cbeta2*cL2; Y2 = a*cbeta2*sL2; Z2 = b*sbeta2
    c1 = Y1*(a/b) - Y1*(b/a) - Y2*(a/b) + Y2*(b/a)
    c2 = -X1*(a/b) + X1*(b/a) + X2*(a/b) - X2*(b/a)
    c3 = (Y1/a)*Z2 - (Y2/a)*Z1
    c4 = (-X1/a)*Z2 + (X2/a)*Z1
    c5 = (X1/b)*Y2 - (X2/b)*Y1
    return [0, c1, c2, c3, c4, c5]

def g_implicita_align_(elli, cs, beta:float, longitude:float):
    a = elli.a; b = elli.b
    sbeta, cbeta = sincos(beta)
    sL, cL = sincos(longitude)
    return cs[1]*cbeta*sbeta*cL + cs[2]*cbeta*sbeta*sL + cs[3]*cbeta*cL + cs[4]*cbeta*sL + cs[5]*sbeta

def g_partial_align_(elli, cs, beta:float, longitude:float):
    c2beta = math.cos(2.0*beta)
    sbeta, cbeta = sincos(beta)
    sL, cL = sincos(longitude)
    g_beta = c2beta*(cs[1]*cL + cs[2]*sL) + sbeta*(-cs[3]*cL - cs[4]*sL) + cs[5]*cbeta
    g_L = cbeta*(sbeta*(cs[2]*cL - cs[1]*sL) - cs[3]*sL + cs[4]*cL)
    return g_beta, g_L

def L2beta_align(elli, cs, beta, L):
    c2beta = math.cos(2.0*beta)
    sbeta, cbeta = sincos(beta)
    for _ in range(30):
        sL, cL = sincos(L)
        g_beta = c2beta*(cs[1]*cL + cs[2]*sL) + sbeta*(-cs[3]*cL - cs[4]*sL) + cs[5]*cbeta
        g_L = cbeta*(sbeta*(cs[2]*cL - cs[1]*sL) - cs[3]*sL + cs[4]*cL)
        g = cs[1]*cbeta*sbeta*cL + cs[2]*cbeta*sbeta*sL + cs[3]*cbeta*cL + cs[4]*cbeta*sL + cs[5]*sbeta
        deltaBeta = g/(g_beta + sgn(g_beta)*EPSILON_MAQ)
        if abs(deltaBeta) > MAXSTEP_ANG:
            deltaBeta = math.copysign(MAXSTEP_ANG, deltaBeta)
        beta = beta - deltaBeta
        c2beta = math.cos(2.0*beta)
        sbeta, cbeta = sincos(beta)
        if math.fabs(deltaBeta) < EPSILON_ANG:
            break
        beta = math.asin(sbeta)
    return beta

def beta2L_align(elli, cs, beta, L0):
    L = L0
    sL, cL = sincos(L)
    for _ in range(30):
        c2beta = math.cos(2.0*beta)
        sbeta, cbeta = sincos(beta)
        g_beta = c2beta*(cs[1]*cL + cs[2]*sL) + sbeta*(-cs[3]*cL - cs[4]*sL) + cs[5]*cbeta
        g_L = cbeta*(sbeta*(cs[2]*cL - cs[1]*sL) - cs[3]*sL + cs[4]*cL)
        g = cs[1]*cbeta*sbeta*cL + cs[2]*cbeta*sbeta*sL + cs[3]*cbeta*cL + cs[4]*cbeta*sL + cs[5]*sbeta
        deltaL = g/(g_L + sgn(g_L)*EPSILON_MAQ)
        if abs(deltaL) > MAXSTEP_ANG:
            deltaL = math.copysign(MAXSTEP_ANG, deltaL)        
        L = L - deltaL
        sL, cL = sincos(L)
        if math.fabs(deltaL) < EPSILON_ANG:
            break
        L = arg(cL, sL)
    return L

def acimut_align_(elli, cs, beta:float, longitude:float):
    e2 = elli.e2; cbeta = math.cos(beta)
    g_beta, g_L = g_partial_align_(elli, cs, beta, longitude)
    y_tanalpha = cbeta*g_beta
    x_tanalpha = -math.sqrt(1.0 - e2*cbeta*cbeta)*g_L
    return arg(x_tanalpha, y_tanalpha)

def edos_L_dist_align_(plat, elli, cs, y, L):    
    beta, s = y
    c2beta = math.cos(2.0*beta)
    sbeta, cbeta = sincos(beta)
    sL, cL = sincos(L)
    #g_beta, g_L = g_partial_(elli, cs, beta, L)
    # --- g_partial_ ---------------------
    g_beta = c2beta*(cs[1]*cL + cs[2]*sL) + sbeta*(-cs[3]*cL - cs[4]*sL) + cs[5]*cbeta
    g_L = cbeta*(sbeta*(cs[2]*cL - cs[1]*sL) - cs[3]*sL + cs[4]*cL)
    # ------------------------------------
    dbeta_dL = -g_L / g_beta
    ds_dL = elli.a*math.sqrt((1 - elli.e2*cbeta**2)*dbeta_dL**2 + cbeta**2)    
    return np.array([dbeta_dL, ds_dL])

def edos_L_area_align_(plat, elli, cs, y, L):
    beta, s, S = y
    c2beta = math.cos(2.0 * beta)
    sbeta, cbeta = sincos(beta)
    sL, cL = sincos(L)
    g_beta = (c2beta * (cs[1] * cL + cs[2] * sL) + sbeta * (-cs[3] * cL - cs[4] * sL) + cs[5] * cbeta)
    g_L = (cbeta*(sbeta * (cs[2] * cL - cs[1] * sL) - cs[3] * sL + cs[4] * cL))
    dbeta_dL = -g_L / g_beta
    ds_dL = elli.a * math.sqrt((1.0 - elli.e2 * cbeta**2) * dbeta_dL**2 + cbeta**2)    
    R_tau_sq = elli.R2**2
    tau = lat2latn6(plat.PARA2AUTH, beta)
    #en caso de que el aplastamiento no es un valor pequeño, se debe calcular en forma exacta
    #aunque en este caso hay una fórmula más directa para calcular la superficie
    #tau = beta2tau(elli, beta)
    
    dS_dL = R_tau_sq * math.sin(tau)

    return np.array([dbeta_dL, ds_dL, dS_dL])


def edos_beta_area_align_(plat, elli, cs, y, h, beta):
    L, s, S = y
    c2beta = math.cos(2.0 * beta)
    sbeta, cbeta = sincos(beta)
    sL, cL = sincos(L)
    g_beta = (c2beta * (cs[1] * cL + cs[2] * sL) + sbeta * (-cs[3] * cL - cs[4] * sL) + cs[5] * cbeta)
    g_L = (cbeta*(sbeta * (cs[2] * cL - cs[1] * sL) - cs[3] * sL + cs[4] * cL))
    dL_dbeta = -g_beta / g_L
    ds_dbeta = elli.a * math.sqrt((1.0 - elli.e2 * cbeta**2) + cbeta**2 * dL_dbeta**2)    
    R_tau_sq = elli.R2**2
    tau = lat2latn6(plat.PARA2AUTH, beta)
    #tau = beta2tau(elli, beta)
    dS_dbeta = R_tau_sq * math.sin(tau) * dL_dbeta
    
    return np.array([dL_dbeta, ds_dbeta, dS_dbeta])

def edos_beta_dist_align_(plat, elli, cs, y, h, beta):
    L, s = y
    c2beta = math.cos(2.0*beta)
    sbeta, cbeta = sincos(beta)
    sL, cL = sincos(L)
    g_beta = c2beta*(cs[1]*cL + cs[2]*sL) + sbeta*(-cs[3]*cL - cs[4]*sL) + cs[5]*cbeta
    g_L = cbeta*(sbeta*(cs[2]*cL - cs[1]*sL) - cs[3]*sL + cs[4]*cL)
    dL_dbeta = -g_beta / g_L
    ds_dbeta = elli.a * math.sqrt((1 - elli.e2*cbeta**2) + cbeta**2 * dL_dbeta**2)
    return np.array([dL_dbeta, ds_dbeta])

def rk45_step_L_area_(edo, h, plat, elli, cs, y, L):
    k1 = edo(plat, elli, cs, y, L)
    k2 = edo(plat, elli, cs, y + h * B[1, 0] * k1, L + A[1] * h)
    k3 = edo(plat, elli, cs, y + h * (B[2, 0] * k1 + B[2, 1] * k2), L + A[2] * h)
    k4 = edo(plat, elli, cs, y + h * (B[3, 0] * k1 + B[3, 1] * k2 + B[3, 2] * k3), L + A[3] * h)
    k5 = edo(plat, elli, cs, y + h * (B[4, 0] * k1 + B[4, 1] * k2 + B[4, 2] * k3 + B[4, 3] * k4), L + A[4] * h)
    k6 = edo(plat, elli, cs, y + h * (B[5, 0] * k1 + B[5, 1] * k2 + B[5, 2] * k3 + B[5, 3] * k4 + B[5, 4] * k5), L + A[5] * h)    
    y_nuevo5 = y + h * (C5[0] * k1 + C5[2] * k3 + C5[3] * k4 + C5[4] * k5 + C5[5] * k6)
    k7 = edo(plat, elli, cs, y_nuevo5, L + A[6] * h)    
    y_nuevo4 = y + h * (C4[0] * k1 + C4[2] * k3 + C4[3] * k4 + C4[4] * k5 + C4[5] * k6 + C4[6] * k7)    
    error = np.abs(y_nuevo5[1] - y_nuevo4[1])
    return y_nuevo5, error

def rk45_step_L_dist_(edo, h, plat, elli, cs, y, L):
    k1 = edo(plat, elli, cs, y, L)
    k2 = edo(plat, elli, cs, y + h*B[1, 0]*k1, L + A[1]*h)
    k3 = edo(plat, elli, cs, y + h*(B[2,0]*k1 + B[2,1]*k2), L + A[2]*h)
    k4 = edo(plat, elli, cs, y + h*(B[3,0]*k1 + B[3,1]*k2 + B[3,2]*k3), L + A[3]*h)
    k5 = edo(plat, elli, cs, y + h*(B[4,0]*k1 + B[4,1]*k2 + B[4,2]*k3 + B[4,3]*k4), L + A[4]*h)
    k6 = edo(plat, elli, cs, y + h*(B[5,0]*k1 + B[5,1]*k2 + B[5,2]*k3 + B[5,3]*k4 + B[5,4]*k5), L + A[5]*h)
    y_nuevo5 = y + h * (C5[0]*k1 + C5[2]*k3 + C5[3]*k4 + C5[4]*k5 + C5[5]*k6)
    k7 = edo(plat, elli, cs, y_nuevo5, L + A[6]*h)
    y_nuevo4 = y + h * (C4[0]*k1 + C4[2]*k3 + C4[3]*k4 + C4[4]*k5 + C4[5]*k6 + C4[6]*k7)
    error = np.abs(y_nuevo5[1] - y_nuevo4[1])
    return y_nuevo5, error

def rk45_step_beta_area_(edo, h, plat, elli, cs, y, beta):
    k1 = edo(plat, elli, cs, y, h, beta)
    k2 = edo(plat, elli, cs, y + h*B[1, 0]*k1, h, beta + A[1]*h)
    k3 = edo(plat, elli, cs, y + h*(B[2,0]*k1 + B[2,1]*k2),h, beta + A[2]*h)
    k4 = edo(plat, elli, cs, y + h*(B[3,0]*k1 + B[3,1]*k2 + B[3,2]*k3),h, beta + A[3]*h)
    k5 = edo(plat, elli, cs, y + h*(B[4,0]*k1 + B[4,1]*k2 + B[4,2]*k3 + B[4,3]*k4),h, beta + A[4]*h)
    k6 = edo(plat, elli, cs, y + h*(B[5,0]*k1 + B[5,1]*k2 + B[5,2]*k3 + B[5,3]*k4 + B[5,4]*k5),h, beta + A[5]*h)
    y_nuevo5 = y + h * (C5[0]*k1 + C5[2]*k3 + C5[3]*k4 + C5[4]*k5 + C5[5]*k6)
    k7 = edo(plat, elli, cs, y_nuevo5, h,beta + A[6]*h)
    y_nuevo4 = y + h * (C4[0]*k1 + C4[2]*k3 + C4[3]*k4 + C4[4]*k5 + C4[5]*k6 + C4[6]*k7)
    error = np.abs(y_nuevo5[1] - y_nuevo4[1])
    return y_nuevo5, error

def rk45_step_beta_dist_(edo, h, plat, elli, cs, y, beta):
    k1 = edo(plat, elli, cs, y, h, beta)
    k2 = edo(plat, elli, cs, y + h*B[1, 0]*k1, h, beta + A[1]*h)
    k3 = edo(plat, elli, cs, y + h*(B[2,0]*k1 + B[2,1]*k2),h, beta + A[2]*h)
    k4 = edo(plat, elli, cs, y + h*(B[3,0]*k1 + B[3,1]*k2 + B[3,2]*k3),h, beta + A[3]*h)
    k5 = edo(plat, elli, cs, y + h*(B[4,0]*k1 + B[4,1]*k2 + B[4,2]*k3 + B[4,3]*k4),h, beta + A[4]*h)
    k6 = edo(plat, elli, cs, y + h*(B[5,0]*k1 + B[5,1]*k2 + B[5,2]*k3 + B[5,3]*k4 + B[5,4]*k5),h, beta + A[5]*h)
    y_nuevo5 = y + h * (C5[0]*k1 + C5[2]*k3 + C5[3]*k4 + C5[4]*k5 + C5[5]*k6)
    k7 = edo(plat, elli, cs, y_nuevo5, h,beta + A[6]*h)
    y_nuevo4 = y + h * (C4[0]*k1 + C4[2]*k3 + C4[3]*k4 + C4[4]*k5 + C4[5]*k6 + C4[6]*k7)
    error = np.abs(y_nuevo5[1] - y_nuevo4[1])
    return y_nuevo5, error

def resolver_dist_L_(tol, h_max, h_min, plat, elli, cs, beta0, L0, LE, edos_L_dist_):
    deltaL_total = tosemicirc(LE - L0)
    sgnL = math.copysign(1.0, deltaL_total)
    deltaL_total = math.fabs(deltaL_total)
    y = np.array([beta0, 0.0])
    L = L0
    avanzado = 0.0
    h = min(h_max, deltaL_total)
    lambdas = [L0]
    betas = [beta0]
    dists = [0.0]
    while avanzado < deltaL_total - EPSILON_ANG:
        if avanzado + h > deltaL_total:
            h = deltaL_total - avanzado

        h_signed = sgnL * h
        y_nuevo, error = rk45_step_L_dist_(edos_L_dist_, h_signed, plat, elli, cs, y, L)

        IDIST = min(MAXDIST, abs(y_nuevo[1] - y[1]))
        err_norm = np.max(error / max(IDIST, EPSILON_ANG))

        if err_norm < tol or h <= h_min:
            y = y_nuevo
            L = tosemicirc(L + h_signed)
            avanzado += h
            lambdas.append(L)
            betas.append(y[0])
            dists.append(abs(y[1]))

        if err_norm == 0:
            factor = 5.0
        else:
            factor = (tol / (err_norm + EPSILON_ANG)) ** (1.0 / 5.0)

        h = min(h_max, max(h_min, h * factor))

    return abs(y[1]), np.array(lambdas), np.array(betas), np.array(dists)


def resolver_area_L_(tol, h_max, h_min, plat, elli, cs, beta0, L0, LE, edos_L_area_):
    deltaL_total = tosemicirc(LE - L0)
    sgnL = math.copysign(1.0, deltaL_total)
    deltaL_total = abs(deltaL_total)
    y = np.array([beta0, 0.0, 0.0])
    L = L0
    avanzado = 0.0
    h = min(h_max, deltaL_total)
    lambdas = [L0]
    betas = [beta0]
    dists = [0.0]
    sups = [0.0]
    while avanzado < deltaL_total - EPSILON_ANG:
        if avanzado + h > deltaL_total:
            h = deltaL_total - avanzado

        h_signed = sgnL * h
        y_nuevo, error = rk45_step_L_area_(edos_L_area_, h_signed, plat, elli, cs, y, L)

        IDIST = min(MAXDIST, abs(y_nuevo[1] - y[1]))
        err_norm = np.max(error / max(IDIST, EPSILON_ANG))

        if err_norm < tol or h <= h_min:
            y = y_nuevo
            L = tosemicirc(L + h_signed)
            avanzado += h
            lambdas.append(L)
            betas.append(y[0])
            dists.append(abs(y[1]))
            sups.append(y[2])

        if err_norm == 0:
            factor = 5.0
        else:
            factor = (tol / (err_norm + EPSILON_ANG)) ** (1.0 / 5.0)

        h = min(h_max, max(h_min, h * factor))

    return (abs(y[1]), y[2], np.array(lambdas), np.array(betas), np.array(dists), np.array(sups))

def resolver_area_beta_(tol, h_max, h_min, plat, elli, cs, L0, beta0, betaE, edos_beta_area_):
    deltaBeta_total = betaE - beta0
    sgnB = math.copysign(1.0, deltaBeta_total)
    deltaBeta_total = abs(deltaBeta_total)
    y = np.array([L0, 0.0, 0.0])
    beta = beta0
    avanzado = 0.0
    h = min(h_max, deltaBeta_total)
    betas = [beta0]
    lambdas = [L0]
    dists = [0.0]
    sups = [0.0]
    while avanzado < deltaBeta_total - EPSILON_ANG:
        if avanzado + h > deltaBeta_total:
            h = deltaBeta_total - avanzado

        h_signed = sgnB * h
        y_nuevo, error = rk45_step_beta_area_(edos_beta_area_, h_signed, plat, elli, cs, y, beta)
        IDIST = min(MAXDIST, abs(y_nuevo[1] - y[1]))
        err_norm = np.max(error / max(IDIST, EPSILON_ANG))
        if err_norm < tol or h <= h_min:
            y = y_nuevo
            beta += h_signed
            avanzado += h
            lambdas.append(tosemicirc(y[0]))
            betas.append(beta)
            dists.append(abs(y[1]))
            sups.append(y[2])

        if err_norm == 0:
            factor = 5.0
        else:
            factor = (tol / (err_norm + EPSILON_ANG)) ** (1.0 / 5.0)

        h = min(h_max, max(h_min, h * factor))

    return (abs(y[1]), y[2], np.array(lambdas), np.array(betas), np.array(dists), np.array(sups))

def resolver_dist_beta_(tol, h_max, h_min, plat, elli, cs, L0, beta0, betaE, edos_beta_dist_):
    deltaBeta_total = betaE - beta0
    sgnB = math.copysign(1.0, deltaBeta_total)
    deltaBeta_total = abs(deltaBeta_total)
    y = np.array([L0, 0.0])
    beta = beta0
    avanzado = 0.0
    h = min(h_max, deltaBeta_total)
    betas = [beta0]
    lambdas = [L0]
    dists = [0.0]
    while avanzado < deltaBeta_total - EPSILON_ANG:
        if avanzado + h > deltaBeta_total:
            h = deltaBeta_total - avanzado

        h_signed = sgnB * h
        y_nuevo, error = rk45_step_beta_dist_(edos_beta_dist_, h_signed, plat, elli, cs, y, beta)
        IDIST = min(MAXDIST, abs(y_nuevo[1] - y[1]))
        err_norm = np.max(error / max(IDIST, EPSILON_ANG))
        if err_norm < tol or h <= h_min:
            y = y_nuevo
            beta += h_signed
            avanzado += h
            lambdas.append(tosemicirc(y[0]))
            betas.append(beta)
            dists.append(abs(y[1]))

        if err_norm == 0:
            factor = 5.0
        else:
            factor = (tol / (err_norm + EPSILON_ANG)) ** (1.0 / 5.0)

        h = min(h_max, max(h_min, h * factor))

    return abs(y[1]), np.array(lambdas), np.array(betas), np.array(dists)

def inverse_area_align_(pRK45, plat, elli, cs, beta1:float, L1:float, beta2:float, L2:float):
    alpha = acimut_align_(elli, cs, beta1, L1)
    salpha_limit = math.fabs(math.sin(math.pi/12.0))
    if (math.fabs(math.sin(alpha)) < salpha_limit) and (math.fabs(L2-L1) < PI_2):
        dist, area, lambdas, betas, dists, areas = resolver_area_beta_(pRK45.tol, pRK45.h_max, pRK45.h_min, plat, elli, cs, L1, beta1, beta2, edos_beta_area_align_)
    else:    
        dist, area, lambdas, betas, dists, areas = resolver_area_L_(pRK45.tol, pRK45.h_max, pRK45.h_min, plat, elli, cs, beta1, L1, L2, edos_L_area_align_)
    return alpha, dist, area, lambdas, betas, dists, areas

def inverse_dist_align_(pRK45, plat, elli, cs, beta1:float, L1:float, beta2:float, L2:float):
    alpha = acimut_align_(elli, cs, beta1, L1)
    salpha_limit = math.fabs(math.sin(math.pi/12.0))
    if (math.fabs(math.sin(alpha)) < salpha_limit) and (math.fabs(L2-L1) < PI_2):
        dist, lambdas, betas, dists = resolver_dist_beta_(pRK45.tol, pRK45.h_max, pRK45.h_min, plat, elli, cs, L1, beta1, beta2, edos_beta_dist_align_)
    else:    
        dist, lambdas, betas, dists = resolver_dist_L_(pRK45.tol, pRK45.h_max, pRK45.h_min, plat, elli, cs, beta1, L1, L2, edos_L_dist_align_)
    return alpha, dist, lambdas, betas, dists

def partials_inverse_dist_align_(pRK45, plat, elli, beta1:float, L1:float, beta2:float, L2:float):
    h = MAXSTEP_h
    cs = cs_align_(elli, beta1, L1, beta2, L2)
    alpha, dist, dummy, dummy, dummy = inverse_dist_align_(pRK45, plat, elli, cs, beta1, L1, beta2, L2)
    cs = cs_align_(elli, beta1, L1, beta2 + h, L2)
    dalpha_beta, ddist_beta, dummy, dummy, dummy = inverse_dist_align_(pRK45, plat, elli, cs, beta1, L1, beta2 + h, L2)
    cs = cs_align_(elli, beta1, L1, beta2, L2 + h)
    dalpha_L, ddist_L, dummy, dummy, dummy = inverse_dist_align_(pRK45, plat, elli, cs, beta1, L1, beta2, L2 + h)

    alpha_beta = (dalpha_beta - alpha)/h
    alpha_L = (dalpha_L - alpha)/h
    dist_beta = (ddist_beta - dist)/h
    dist_L = (ddist_L - dist)/h

    return alpha, dist, alpha_beta, dist_beta, alpha_L, dist_L

def direct_curva_align_(plat, elli, phi1:float, L1:float, ALPHA, DIST):
    phi2, L2 = CentralSect2GEO(plat, elli, phi1, L1, ALPHA, DIST)
    beta1 = phi2beta(elli, phi1)
    beta2 = phi2beta(elli, phi2)
    tol = EPSILON_ANG; h_max = MAXSTEPRK45_ANG; h_min = EPSILON_ANG
    pRK45 = pRK45_t(tol, h_max, h_min)
    cs = cs_align_(elli, beta1, L1, beta2, L2)
    for i in range(1,30):
        alpha, dist, alpha_beta, dist_beta, alpha_L, dist_L = partials_inverse_dist_align_(pRK45, plat, elli, beta1, L1, beta2, L2)
        a = alpha_beta
        b = alpha_L
        c = dist_beta
        d = dist_L        
        F1 = alpha - ALPHA
        F2 = dist - DIST
        det = a*d - b*c
        sgndet = sgn(det)
        deltaBeta = ( d*F1 - b*F2)/(det + sgndet*EPSILON_MAQ)
        deltaL    = (-c*F1 + a*F2)/(det + sgndet*EPSILON_MAQ)
        if abs(deltaL) > MAXSTEP_ANG:
            deltaL = math.copysign(MAXSTEP_ANG, deltaL)
        if abs(deltaBeta) > MAXSTEP_ANG:
            deltaBeta = math.copysign(MAXSTEP_ANG, deltaBeta)
        beta2 = beta2 - deltaBeta
        L2 = L2 - deltaL
        cs = cs_align_(elli, beta1, L1, beta2, L2)
        if DEBUG: print(deltaL)
        if max(math.fabs(deltaBeta), math.fabs(deltaL)) < EPSILON_ANG:
            break
    sbeta2, dummy = sincos(beta2)
    beta2 = math.asin(sbeta2)
    sL2, cL2 = sincos(L2)
    L2 = arg(cL2, sL2)
    return beta2phi(elli, beta2), L2

def inv_align_dist(plat, elli, phi1:float, L1:float, phi2:float, L2:float, L11:float, L22:float):
    alphas = []
    tol = EPSILON_ANG; h_max = MAXSTEPRK45_ANG; h_min = EPSILON_ANG    
    pRK45 = pRK45_t(tol, h_max, h_min)
    beta1 = phi2beta(elli, phi1)
    beta2 = phi2beta(elli, phi2)
    cs = cs_align_(elli, beta1, L1, beta2, L2)
    beta11 = L2beta_align(elli, cs, beta1, L11)
    beta22 = L2beta_align(elli, cs, beta2, L22)
    Beta0, L0 = CalcBeta0_align(elli, cs, beta2, 0.5*(L2-L1))
    alpha, dist, lambdas, betas, dists = inverse_dist_align_(pRK45, plat, elli, cs, beta11, L11, beta22, L22)
    phis = []
    Phi0 = beta2phi(elli, Beta0)
    for i in range(0, len(betas)):
        phis.append(RAD2DEG(beta2phi(elli, betas[i])))
        alphas.append(RAD2DEG(acimut_align_(elli, cs, betas[i], lambdas[i])))
        lambdas[i] = RAD2DEG(lambdas[i])

    pathpoints = np.column_stack((phis, lambdas, alphas, dists))
    return alpha, dist, Phi0, L0, pathpoints

def inv_align_area(plat, elli, phi1:float, L1:float, phi2:float, L2:float, L11:float, L22:float):
    alphas = []
    tol = EPSILON_ANG; h_max = MAXSTEPRK45_ANG; h_min = EPSILON_ANG    
    pRK45 = pRK45_t(tol, h_max, h_min)
    beta1 = phi2beta(elli, phi1)
    beta2 = phi2beta(elli, phi2)
    cs = cs_align_(elli, beta1, L1, beta2, L2)
    beta11 = L2beta_align(elli, cs, beta1, L11)
    beta22 = L2beta_align(elli, cs, beta2, L22)
    Beta0, L0 = CalcBeta0_align(elli, cs, beta2, 0.5*(L2-L1))
    alpha, dist, area, lambdas, betas, dists, areas = inverse_area_align_(pRK45, plat, elli, cs, beta11, L11, beta22, L22)
    phis = []
    Phi0 = beta2phi(elli, Beta0)
    for i in range(0, len(betas)):
        phis.append(RAD2DEG(beta2phi(elli, betas[i])))
        alphas.append(RAD2DEG(acimut_align_(elli, cs, betas[i], lambdas[i])))
        lambdas[i] = RAD2DEG(lambdas[i])

    pathpoints = np.column_stack((phis, lambdas, alphas, dists, areas))
    return alpha, dist, area, Phi0, L0, pathpoints

def CalcBeta0_align(elli, cs, beta, L):
    sbeta, cbeta = sincos(beta)
    sL, cL = sincos(L)
    for _ in range(30):
        c2beta = math.cos(2.0*beta)
        g = cs[1]*cbeta*sbeta*cL + cs[2]*cbeta*sbeta*sL + cs[3]*cbeta*cL + cs[4]*cbeta*sL + cs[5]*sbeta
        g_beta = c2beta*(cs[1]*cL + cs[2]*sL) + sbeta*(-cs[3]*cL - cs[4]*sL) + cs[5]*cbeta
        g_L = cbeta*(sbeta*(cs[2]*cL - cs[1]*sL) - cs[3]*sL + cs[4]*cL)
        g_L_B = 2.0*cbeta**2*(cs[2]*cL - cs[1]*sL) + sbeta*(cs[3]*sL - cs[4]*cL) + cs[1]*sL - cs[2]*cL
        g_L_L = -cbeta*(sbeta*(cs[1]*cL + cs[2]*sL) + cs[3]*cL + cs[4]*sL)
        a = g_beta
        b = g_L
        c = g_L_B
        d = g_L_L
        F1 = g
        F2 = g_L
        det = a*d - b*c
        sgndet = sgn(det)
        deltaBeta = ( d*F1 - b*F2)/(det + sgndet*EPSILON_MAQ)
        deltaL    = (-c*F1 + a*F2)/(det + sgndet*EPSILON_MAQ)
        if abs(deltaL) > MAXSTEP_ANG:
            deltaL = math.copysign(MAXSTEP_ANG, deltaL)

        if abs(deltaBeta) > MAXSTEP_ANG:
            deltaBeta = math.copysign(MAXSTEP_ANG, deltaBeta)

        beta = beta - deltaBeta        
        L = L - deltaL
        sbeta, cbeta = sincos(beta)
        sL, cL = sincos(L)
        if max(math.fabs(deltaBeta), math.fabs(deltaL)) < EPSILON_ANG:
            break

    beta = math.asin(sbeta)
    L = arg(cL, sL)
    return beta, L

def CentralSect2GEO(plat, elli:pelipsoide_t, phi1, L1, alpha, dist):
    f = elli.f
    sinL1, cosL1 = sincos(L1)
    sinphi1, cosphi1 = sincos(phi1)
    beta1 = phi2beta(elli, phi1)     
    sinbeta1, cosbeta1 = sincos(beta1)
    psi1 = phi2psi(elli, phi1)
    sinpsi1, cospsi1 = sincos(psi1)
    rho = dist
    sinalpha, cosalpha = sincos(alpha)

    nx = (1.0-f)*(sinphi1*sinL1*cosalpha-cosL1*sinalpha)*sinbeta1+cosbeta1*cosphi1*sinL1*cosalpha
    ny = (1.0-f)*(-sinphi1*cosL1*cosalpha-sinL1*sinalpha)*sinbeta1-cosbeta1*cosphi1*cosL1*cosalpha
    nz = cosbeta1*sinalpha

    psi0 = arg(nz, math.hypot(nx, ny))
    if beta1<0:
        L0 = arg(ny, -nx)
    else:
        L0 = arg(-ny, nx)
    
    bdot = elli.a/math.sqrt(1.0 + elli.ep2*math.sin(psi0)**2)
    fdot = 1.0 - bdot/elli.a
    ellidot = pelipsoide_init(fdot, elli.a)
    platdot = init_platn6(ellidot.n)
    mudot = rho/ellidot.Rmu
    
    sinLL10 = math.sin(L1 - L0)
    cosLL10 = math.cos(L1 - L0)

    u = math.sqrt((1.0 - elli.e2)*sinbeta1**2 + (cosbeta1*sinLL10)**2)
    v = cosbeta1*cosLL10
    thetadot1 = arg(v, u)
    mudot1 = lat2latn6(platdot.GEOC2RECT, thetadot1)

    mudot2 = mudot + mudot1

    thetadot2 = lat2latn6(platdot.RECT2GEOC, mudot2)
    thetadot = thetadot2 - thetadot1
    
    costhetadot = math.cos(thetadot)
    sinthetadot = math.sin(thetadot)

    ux = cospsi1*cosL1
    uy = cospsi1*sinL1
    uz = sinpsi1

    vx = ny*uz - nz*uy
    vy = -nx*uz + nz*ux
    vz = nx*uy - ny*ux

    vnorma = math.sqrt(vx**2 + vy**2 + vz**2)
    wx = vx/vnorma
    wy = vy/vnorma
    wz = vz/vnorma
    
    rx = ux*costhetadot + wx*sinthetadot
    ry = uy*costhetadot + wy*sinthetadot
    rz = uz*costhetadot + wz*sinthetadot

    #psi = arg(math.hypot(rx,ry), rz)
    phi = arg((1.0-elli.e2)*math.hypot(rx,ry), rz)
    L = arg(rx, ry)
    #phi = psi2phi(elli, psi)

    return phi, L

def cs_central_(elli, beta1:float, L1:float, beta2:float, L2:float):
    a = elli.a; b = elli.b
    sbeta1, cbeta1 = sincos(beta1)
    sL1, cL1 = sincos(L1)    
    sbeta2, cbeta2 = sincos(beta2)
    sL2, cL2 = sincos(L2)
    X1 = a*cbeta1*cL1; Y1 = a*cbeta1*sL1; Z1 = b*sbeta1
    X2 = a*cbeta2*cL2; Y2 = a*cbeta2*sL2; Z2 = b*sbeta2
    c1 = (Y1/b)*Z2 - Y2*(Z1/b)
    c2 = -(X1/b)*Z2 + X2*(Z1/b)
    c3 = (X1/a)*Y2 - X2*(Y1/a)
    return [0, c1, c2, c3]

def g_implicita_central_(elli, cs, beta:float, longitude:float):
    a = elli.a; b = elli.b
    sbeta, cbeta = sincos(beta)
    sL, cL = sincos(longitude)
    return cs[1]*cbeta*cL + cs[2]*cbeta*sL + cs[3]*sbeta

def g_partial_central_(elli, cs, beta:float, longitude:float):
    sbeta, cbeta = sincos(beta)
    sL, cL = sincos(longitude)    
    g_beta = -cs[1]*sbeta*cL - cs[2]*sbeta*sL + cs[3]*cbeta
    g_L = -cs[1]*cbeta*sL + cs[2]*cbeta*cL
    return g_beta, g_L

def L2beta_central(elli, cs, beta, L):
    for _ in range(30):
        sL, cL = sincos(L)
        sbeta, cbeta = sincos(beta)
        g_beta = -cs[1]*sbeta*cL - cs[2]*sbeta*sL + cs[3]*cbeta
        g_L = -cs[1]*cbeta*sL + cs[2]*cbeta*cL
        g = cs[1]*cbeta*cL + cs[2]*cbeta*sL + cs[3]*sbeta
        deltaBeta = g/(g_beta + sgn(g_beta)*EPSILON_MAQ)
        if abs(deltaBeta) > MAXSTEP_ANG:
            deltaBeta = math.copysign(MAXSTEP_ANG, deltaBeta)

        beta = beta - deltaBeta
        if math.fabs(deltaBeta) < EPSILON_ANG:
            break
        sbeta, cbeta = sincos(beta)
        beta = math.asin(sbeta)
    return beta

def beta2L_central(elli, cs, beta, L0):
    L = L0
    sL, cL = sincos(L)
    for _ in range(30):
        sbeta, cbeta = sincos(beta)
        g_beta = -cs[1]*sbeta*cL - cs[2]*sbeta*sL + cs[3]*cbeta
        g_L = -cs[1]*cbeta*sL + cs[2]*cbeta*cL
        g = cs[1]*cbeta*cL + cs[2]*cbeta*sL + cs[3]*sbeta
        deltaL = g/(g_L + sgn(g_L)*EPSILON_MAQ)
        if abs(deltaL) > MAXSTEP_ANG:
            deltaL = math.copysign(MAXSTEP_ANG, deltaL)
        L = L - deltaL        
        sL, cL = sincos(L)
        if math.fabs(deltaL) < EPSILON_ANG:
            break
        L = arg(cL, sL)
    return L

def acimut_central_(elli, cs, beta:float, longitude:float):
    e2 = elli.e2; cbeta = math.cos(beta)
    g_beta, g_L = g_partial_central_(elli, cs, beta, longitude)
    y_tanalpha = cbeta*g_beta
    x_tanalpha = -math.sqrt(1.0 - e2*cbeta*cbeta)*g_L
    return arg(x_tanalpha, y_tanalpha)

def edos_L_area_central_(plat, elli, cs, y, L):
    beta, s, S = y
    sbeta, cbeta = sincos(beta)
    sL, cL = sincos(L)
    g_beta = -cs[1]*sbeta*cL - cs[2]*sbeta*sL + cs[3]*cbeta
    g_L = -cs[1]*cbeta*sL + cs[2]*cbeta*cL
    dbeta_dL = -g_L / g_beta
    ds_dL = elli.a*math.sqrt((1 - elli.e2*cbeta**2)*dbeta_dL**2 + cbeta**2)
    R_tau_sq = elli.R2**2
    tau = lat2latn6(plat.PARA2AUTH, beta)
    dS_dL = R_tau_sq * math.sin(tau)
    #dS_dL = math.fabs(dS_dL)
    return np.array([dbeta_dL, ds_dL, dS_dL])

def edos_L_dist_central_(plat, elli, cs, y, L):    
    beta, s = y
    sbeta, cbeta = sincos(beta)
    sL, cL = sincos(L)
    g_beta = -cs[1]*sbeta*cL - cs[2]*sbeta*sL + cs[3]*cbeta
    g_L = -cs[1]*cbeta*sL + cs[2]*cbeta*cL
    dbeta_dL = -g_L / g_beta
    ds_dL = elli.a*math.sqrt((1 - elli.e2*cbeta**2)*dbeta_dL**2 + cbeta**2)    
    return np.array([dbeta_dL, ds_dL])

def edos_beta_area_central_(plat, elli, cs, y, h, beta):
    L, s, S = y
    sbeta, cbeta = sincos(beta)
    sL, cL = sincos(L)
    g_beta = -cs[1]*sbeta*cL - cs[2]*sbeta*sL + cs[3]*cbeta
    g_L = -cs[1]*cbeta*sL + cs[2]*cbeta*cL
    dL_dbeta = -g_beta / g_L
    ds_dbeta = elli.a * math.sqrt((1 - elli.e2*cbeta**2) + cbeta**2 * dL_dbeta**2)        
    R_tau_sq = elli.R2**2
    tau = lat2latn6(plat.PARA2AUTH, beta)
    dS_dbeta = R_tau_sq * math.sin(tau) * dL_dbeta
    #dS_dbeta = math.fabs(dS_dbeta)
    return np.array([dL_dbeta, ds_dbeta, dS_dbeta])

def edos_beta_dist_central_(plat, elli, cs, y, h, beta):
    L, s = y
    sbeta, cbeta = sincos(beta)
    sL, cL = sincos(L)
    g_beta = -cs[1]*sbeta*cL - cs[2]*sbeta*sL + cs[3]*cbeta
    g_L = -cs[1]*cbeta*sL + cs[2]*cbeta*cL    
    dL_dbeta = -g_beta / g_L
    ds_dbeta = elli.a * math.sqrt((1 - elli.e2*cbeta**2) + cbeta**2 * dL_dbeta**2)
    return np.array([dL_dbeta, ds_dbeta])

def inverse_area_central_(pRK45, plat, elli, cs, beta1:float, L1:float, beta2:float, L2:float):
    alpha = acimut_central_(elli, cs, beta1, L1)
    salpha_limit = math.fabs(math.sin(math.pi/12.0))
    if (math.fabs(math.sin(alpha)) < salpha_limit) and (math.fabs(L2-L1) < PI_2):
        dist, area, lambdas, betas, dists, areas = resolver_area_beta_(pRK45.tol, pRK45.h_max, pRK45.h_min, plat, elli, cs, L1, beta1, beta2, edos_beta_area_central_)
    else:    
        dist, area, lambdas, betas, dists, areas = resolver_area_L_(pRK45.tol, pRK45.h_max, pRK45.h_min, plat, elli, cs, beta1, L1, L2, edos_L_area_central_)
    return alpha, dist, area, lambdas, betas, dists, areas

def inverse_dist_central_(pRK45, plat, elli, cs, beta1:float, L1:float, beta2:float, L2:float):
    alpha = acimut_central_(elli, cs, beta1, L1)
    salpha_limit = math.fabs(math.sin(math.pi/12.0))
    if (math.fabs(math.sin(alpha)) < salpha_limit) and (math.fabs(L2-L1) < PI_2):
        dist, lambdas, betas, dists = resolver_dist_beta_(pRK45.tol, pRK45.h_max, pRK45.h_min, plat, elli, cs, L1, beta1, beta2, edos_beta_dist_central_)
    else:    
        dist, lambdas, betas, dists = resolver_dist_L_(pRK45.tol, pRK45.h_max, pRK45.h_min, plat, elli, cs, beta1, L1, L2, edos_L_dist_central_)
    return alpha, dist, lambdas, betas, dists

def partials_inverse_dist_central_(pRK45, plat, elli, beta1:float, L1:float, beta2:float, L2:float):
    h = MAXSTEP_h
    cs = cs_central_(elli, beta1, L1, beta2, L2)
    alpha, dist, dummy, dummy, dummy = inverse_dist_central_(pRK45, plat, elli, cs, beta1, L1, beta2, L2)
    cs = cs_central_(elli, beta1, L1, beta2 + h, L2)
    dalpha_beta, ddist_beta, dummy, dummy, dummy = inverse_dist_central_(pRK45, plat, elli, cs, beta1, L1, beta2 + h, L2)
    cs = cs_central_(elli, beta1, L1, beta2, L2 + h)
    dalpha_L, ddist_L, dummy, dummy, dummy = inverse_dist_central_(pRK45, plat, elli, cs, beta1, L1, beta2, L2 + h)

    alpha_beta = (dalpha_beta - alpha)/h
    alpha_L = (dalpha_L - alpha)/h
    dist_beta = (ddist_beta - dist)/h
    dist_L = (ddist_L - dist)/h

    return alpha, dist, alpha_beta, dist_beta, alpha_L, dist_L

def direct_curva_central_(plat, elli, phi1:float, L1:float, ALPHA, DIST):
    phi2, L2 = CentralSect2GEO(plat, elli, phi1, L1, ALPHA, DIST)
    beta1 = phi2beta(elli, phi1)
    beta2 = phi2beta(elli, phi2)
    tol = EPSILON_ANG; h_max = MAXSTEPRK45_ANG; h_min = EPSILON_ANG
    pRK45 = pRK45_t(tol, h_max, h_min)
    cs = cs_central_(elli, beta1, L1, beta2, L2)
    for i in range(1,30):
        alpha, dist, alpha_beta, dist_beta, alpha_L, dist_L = partials_inverse_dist_central_(pRK45, plat, elli, beta1, L1, beta2, L2)
        a = alpha_beta
        b = alpha_L
        c = dist_beta
        d = dist_L        
        F1 = alpha - ALPHA
        F2 = dist - DIST
        det = a*d - b*c
        sgndet = sgn(det)
        deltaBeta = ( d*F1 - b*F2)/(det + sgndet*EPSILON_MAQ)
        deltaL    = (-c*F1 + a*F2)/(det + sgndet*EPSILON_MAQ)
        if abs(deltaL) > MAXSTEP_ANG:
            deltaL = math.copysign(MAXSTEP_ANG, deltaL)
        if abs(deltaBeta) > MAXSTEP_ANG:
            deltaBeta = math.copysign(MAXSTEP_ANG, deltaBeta)
        beta2 = beta2 - deltaBeta
        L2 = L2 - deltaL
        cs = cs_central_(elli, beta1, L1, beta2, L2)
        if DEBUG: print(deltaL)
        if max(math.fabs(deltaBeta), math.fabs(deltaL)) < EPSILON_ANG:
            break
    sbeta2, dummy = sincos(beta2)
    beta2 = math.asin(sbeta2)
    sL2, cL2 = sincos(L2)
    L2 = arg(cL2, sL2)
    return beta2phi(elli, beta2), L2

def inv_central_dist(plat, elli, phi1:float, L1:float, phi2:float, L2:float, L11:float, L22:float):
    alphas = []
    tol = EPSILON_ANG; h_max = MAXSTEPRK45_ANG; h_min = EPSILON_ANG    
    pRK45 = pRK45_t(tol, h_max, h_min)
    beta1 = phi2beta(elli, phi1)
    beta2 = phi2beta(elli, phi2)
    cs = cs_central_(elli, beta1, L1, beta2, L2)
    beta11 = L2beta_central(elli, cs, beta1, L11)
    beta22 = L2beta_central(elli, cs, beta2, L22)
    Beta0, L0 = CalcBeta0_central(elli, cs, beta2, 0.5*(L2-L1))
    alpha, dist, lambdas, betas, dists = inverse_dist_central_(pRK45, plat, elli, cs, beta11, L11, beta22, L22)
    phis = []
    Phi0 = beta2phi(elli, Beta0)
    for i in range(0, len(betas)):
        phis.append(RAD2DEG(beta2phi(elli, betas[i])))
        alphas.append(RAD2DEG(acimut_central_(elli, cs, betas[i], lambdas[i])))
        lambdas[i] = RAD2DEG(lambdas[i])

    pathpoints = np.column_stack((phis, lambdas, alphas, dists))
    return alpha, dist, Phi0, L0, pathpoints

def inv_central_area(plat, elli, phi1:float, L1:float, phi2:float, L2:float, L11:float, L22:float):
    alphas = []
    tol = EPSILON_ANG; h_max = MAXSTEPRK45_ANG; h_min = EPSILON_ANG    
    pRK45 = pRK45_t(tol, h_max, h_min)
    beta1 = phi2beta(elli, phi1)
    beta2 = phi2beta(elli, phi2)
    cs = cs_central_(elli, beta1, L1, beta2, L2)
    beta11 = L2beta_central(elli, cs, beta1, L11)
    beta22 = L2beta_central(elli, cs, beta2, L22)
    Beta0, L0 = CalcBeta0_central(elli, cs, beta2, 0.5*(L2-L1))
    alpha, dist, area, lambdas, betas, dists, areas = inverse_area_central_(pRK45, plat, elli, cs, beta11, L11, beta22, L22)
    phis = []
    Phi0 = beta2phi(elli, Beta0)
    for i in range(0, len(betas)):
        phis.append(RAD2DEG(beta2phi(elli, betas[i])))
        alphas.append(RAD2DEG(acimut_central_(elli, cs, betas[i], lambdas[i])))
        lambdas[i] = RAD2DEG(lambdas[i])

    pathpoints = np.column_stack((phis, lambdas, alphas, dists, areas))
    return alpha, dist, area, Phi0, L0, pathpoints


def beta_central(plat, elli, phi1:float, L1:float, phi2:float, L2:float, L):
    alphas = []
    tol = EPSILON_ANG; h_max = MAXSTEPRK45_ANG; h_min = EPSILON_ANG    
    pRK45 = pRK45_t(tol, h_max, h_min)
    beta1 = phi2beta(elli, phi1)
    beta2 = phi2beta(elli, phi2)
    cs = cs_central_(elli, beta1, L1, beta2, L2)
    sL, cL = sincos(L)
    D = cs[1]*cL + cs[2]*sL
    sgnacos = sgn(L2 - L1)
    beta = math.atan(-D/cs[3])
    beta = math.asin(math.sin(beta))
    return beta

def CalcBeta0_central(elli, cs, beta, L):
    sbeta, cbeta = sincos(beta)
    sL, cL = sincos(L)
    for _ in range(30):
        g_beta = -cs[1]*sbeta*cL - cs[2]*sbeta*sL + cs[3]*cbeta
        g_L = -cs[1]*cbeta*sL + cs[2]*cbeta*cL
        g = cs[1]*cbeta*cL + cs[2]*cbeta*sL + cs[3]*sbeta
        g_L_B = cs[1]*sbeta*sL - cs[2]*sbeta*cL
        g_L_L = -cs[1]*cbeta*cL - cs[2]*cbeta*sL
        a = g_beta
        b = g_L
        c = g_L_B
        d = g_L_L
        F1 = g
        F2 = g_L
        det = a*d - b*c
        sgndet = sgn(det)
        deltaBeta = ( d*F1 - b*F2)/(det + sgndet*EPSILON_MAQ)
        deltaL    = (-c*F1 + a*F2)/(det + sgndet*EPSILON_MAQ)
        if abs(deltaL) > MAXSTEP_ANG:
            deltaL = math.copysign(MAXSTEP_ANG, deltaL)

        if abs(deltaBeta) > MAXSTEP_ANG:
            deltaBeta = math.copysign(MAXSTEP_ANG, deltaBeta)

        beta = beta - deltaBeta        
        L = L - deltaL
        sbeta, cbeta = sincos(beta)
        sL, cL = sincos(L)
        if max(math.fabs(deltaBeta), math.fabs(deltaL)) < EPSILON_ANG:
            break

    beta = math.asin(sbeta)
    L = arg(cL, sL)
    return beta, L

def cs_normal_(elli, beta1:float, L1:float, beta2:float, L2:float):
    a = elli.a; b = elli.b
    sbeta1, cbeta1 = sincos(beta1)
    sL1, cL1 = sincos(L1)    
    sbeta2, cbeta2 = sincos(beta2)
    sL2, cL2 = sincos(L2)
    X1 = a*cbeta1*cL1; Y1 = a*cbeta1*sL1; Z1 = b*sbeta1
    X2 = a*cbeta2*cL2; Y2 = a*cbeta2*sL2; Z2 = b*sbeta2
    c1 = a*(Y1/b)*(Z1/b) - (Y1/a)*Z1 + (Y1/a)*Z2 - a*(Y2/b)*(Z1/b)
    c2 = -a*(X1/b)*(Z1/b) + (X1/a)*Z1 - (X1/a)*Z2 + a*(X2/b)*(Z1/b)
    c3 = b*(X1/a)*(Y2/a) - b*(X2/a)*(Y1/a)
    c4 = (X1/b)*(Y2/b)*Z1 - X1*(Y2/a)*(Z1/a) - (X2/b)*Y1*(Z1/b) + X2*(Y1/a)*(Z1/a)
    return [0, c1, c2, c3, c4]

def g_implicita_normal_(elli, cs, beta:float, longitude:float):
    a = elli.a; b = elli.b
    sbeta, cbeta = sincos(beta)
    sL, cL = sincos(longitude)
    return cs[1]*cbeta*cL + cs[2]*cbeta*sL + cs[3]*sbeta + cs[4]

def g_partial_normal_(elli, cs, beta:float, longitude:float):
    sbeta, cbeta = sincos(beta)
    sL, cL = sincos(longitude)    
    g_beta = -cs[1]*sbeta*cL - cs[2]*sbeta*sL + cs[3]*cbeta
    g_L = -cs[1]*cbeta*sL + cs[2]*cbeta*cL
    return g_beta, g_L

def L2beta_normal(elli, cs, beta, L):
    for _ in range(30):
        sL, cL = sincos(L)
        sbeta, cbeta = sincos(beta)
        g_beta = -cs[1]*sbeta*cL - cs[2]*sbeta*sL + cs[3]*cbeta
        g_L = -cs[1]*cbeta*sL + cs[2]*cbeta*cL
        g = cs[1]*cbeta*cL + cs[2]*cbeta*sL + cs[3]*sbeta + cs[4]
        deltaBeta = g/(g_beta + sgn(g_beta)*EPSILON_MAQ)
        if abs(deltaBeta) > MAXSTEP_ANG:
            deltaBeta = math.copysign(MAXSTEP_ANG, deltaBeta)

        beta = beta - deltaBeta
        if math.fabs(deltaBeta) < EPSILON_ANG:
            break
        sbeta, cbeta = sincos(beta)
        beta = math.asin(sbeta)
    return beta

def beta2L_normal(elli, cs, beta, L0):
    L = L0
    sL, cL = sincos(L)
    for _ in range(30):
        sbeta, cbeta = sincos(beta)
        g_beta = -cs[1]*sbeta*cL - cs[2]*sbeta*sL + cs[3]*cbeta
        g_L = -cs[1]*cbeta*sL + cs[2]*cbeta*cL
        g = cs[1]*cbeta*cL + cs[2]*cbeta*sL + cs[3]*sbeta + cs[4]
        deltaL = g/(g_L + sgn(g_L)*EPSILON_MAQ)
        if abs(deltaL) > MAXSTEP_ANG:
            deltaL = math.copysign(MAXSTEP_ANG, deltaL)
                
        L = L - deltaL
        sL, cL = sincos(L)
        if math.fabs(deltaL) < EPSILON_ANG:
            break
        L = arg(cL, sL)
    return L

def acimut_normal_(elli, cs, beta:float, longitude:float):
    e2 = elli.e2; cbeta = math.cos(beta)
    g_beta, g_L = g_partial_normal_(elli, cs, beta, longitude)
    y_tanalpha = cbeta*g_beta
    x_tanalpha = -math.sqrt(1.0 - e2*cbeta*cbeta)*g_L
    return arg(x_tanalpha, y_tanalpha)

def edos_L_area_normal_(plat, elli, cs, y, L):
    beta, s, S = y
    sbeta, cbeta = sincos(beta)
    sL, cL = sincos(L)
    g_beta = -cs[1]*sbeta*cL - cs[2]*sbeta*sL + cs[3]*cbeta
    g_L = -cs[1]*cbeta*sL + cs[2]*cbeta*cL
    dbeta_dL = -g_L / g_beta
    ds_dL = elli.a*math.sqrt((1 - elli.e2*cbeta**2)*dbeta_dL**2 + cbeta**2)
    R_tau_sq = elli.R2**2
    tau = lat2latn6(plat.PARA2AUTH, beta)
    dS_dL = R_tau_sq * math.sin(tau)
    #dS_dL = math.fabs(dS_dL)
    return np.array([dbeta_dL, ds_dL, dS_dL])

def edos_L_dist_normal_(plat, elli, cs, y, L):    
    beta, s = y
    sbeta, cbeta = sincos(beta)
    sL, cL = sincos(L)
    g_beta = -cs[1]*sbeta*cL - cs[2]*sbeta*sL + cs[3]*cbeta
    g_L = -cs[1]*cbeta*sL + cs[2]*cbeta*cL
    dbeta_dL = -g_L / g_beta
    ds_dL = elli.a*math.sqrt((1 - elli.e2*cbeta**2)*dbeta_dL**2 + cbeta**2)    
    return np.array([dbeta_dL, ds_dL])

def edos_beta_area_normal_(plat, elli, cs, y, h, beta):
    L, s, S = y
    sbeta, cbeta = sincos(beta)
    sL, cL = sincos(L)
    g_beta = -cs[1]*sbeta*cL - cs[2]*sbeta*sL + cs[3]*cbeta
    g_L = -cs[1]*cbeta*sL + cs[2]*cbeta*cL
    dL_dbeta = -g_beta / g_L
    ds_dbeta = elli.a * math.sqrt((1 - elli.e2*cbeta**2) + cbeta**2 * dL_dbeta**2)        
    R_tau_sq = elli.R2**2
    tau = lat2latn6(plat.PARA2AUTH, beta)
    dS_dbeta = R_tau_sq * math.sin(tau) * dL_dbeta
    #dS_dbeta = math.fabs(dS_dbeta)
    return np.array([dL_dbeta, ds_dbeta, dS_dbeta])

def edos_beta_dist_normal_(plat, elli, cs, y, h, beta):
    L, s = y
    sbeta, cbeta = sincos(beta)
    sL, cL = sincos(L)
    g_beta = -cs[1]*sbeta*cL - cs[2]*sbeta*sL + cs[3]*cbeta
    g_L = -cs[1]*cbeta*sL + cs[2]*cbeta*cL    
    dL_dbeta = -g_beta / g_L
    ds_dbeta = elli.a * math.sqrt((1 - elli.e2*cbeta**2) + cbeta**2 * dL_dbeta**2)
    return np.array([dL_dbeta, ds_dbeta])

def inverse_area_normal_(pRK45, plat, elli, cs, beta1:float, L1:float, beta2:float, L2:float):
    alpha = acimut_normal_(elli, cs, beta1, L1)
    salpha_limit = math.fabs(math.sin(math.pi/12.0))
    if (math.fabs(math.sin(alpha)) < salpha_limit) and (math.fabs(L2-L1) < PI_2):
        dist, area, lambdas, betas, dists, areas = resolver_area_beta_(pRK45.tol, pRK45.h_max, pRK45.h_min, plat, elli, cs, L1, beta1, beta2, edos_beta_area_normal_)
    else:    
        dist, area, lambdas, betas, dists, areas = resolver_area_L_(pRK45.tol, pRK45.h_max, pRK45.h_min, plat, elli, cs, beta1, L1, L2, edos_L_area_normal_)
    return alpha, dist, area, lambdas, betas, dists, areas

def inverse_dist_normal_(pRK45, plat, elli, cs, beta1:float, L1:float, beta2:float, L2:float):
    alpha = acimut_normal_(elli, cs, beta1, L1)
    salpha_limit = math.fabs(math.sin(math.pi/12.0))
    if (math.fabs(math.sin(alpha)) < salpha_limit) and (math.fabs(L2-L1) < PI_2):
        dist, lambdas, betas, dists = resolver_dist_beta_(pRK45.tol, pRK45.h_max, pRK45.h_min, plat, elli, cs, L1, beta1, beta2, edos_beta_dist_normal_)
    else:    
        dist, lambdas, betas, dists = resolver_dist_L_(pRK45.tol, pRK45.h_max, pRK45.h_min, plat, elli, cs, beta1, L1, L2, edos_L_dist_normal_)
    return alpha, dist, lambdas, betas, dists

def partials_inverse_dist_normal_(pRK45, plat, elli, beta1:float, L1:float, beta2:float, L2:float):
    h = MAXSTEP_h
    cs = cs_normal_(elli, beta1, L1, beta2, L2)
    alpha, dist, dummy, dummy, dummy = inverse_dist_normal_(pRK45, plat, elli, cs, beta1, L1, beta2, L2)
    cs = cs_normal_(elli, beta1, L1, beta2 + h, L2)
    dalpha_beta, ddist_beta, dummy, dummy, dummy = inverse_dist_normal_(pRK45, plat, elli, cs, beta1, L1, beta2 + h, L2)
    cs = cs_normal_(elli, beta1, L1, beta2, L2 + h)
    dalpha_L, ddist_L, dummy, dummy, dummy = inverse_dist_normal_(pRK45, plat, elli, cs, beta1, L1, beta2, L2 + h)

    alpha_beta = (dalpha_beta - alpha)/h
    alpha_L = (dalpha_L - alpha)/h
    dist_beta = (ddist_beta - dist)/h
    dist_L = (ddist_L - dist)/h

    return alpha, dist, alpha_beta, dist_beta, alpha_L, dist_L

def direct_curva_normal_(plat, elli, phi1:float, L1:float, ALPHA, DIST):
    phi2, L2 = CentralSect2GEO(plat, elli, phi1, L1, ALPHA, DIST)
    beta1 = phi2beta(elli, phi1)
    beta2 = phi2beta(elli, phi2)
    tol = EPSILON_ANG; h_max = MAXSTEPRK45_ANG; h_min = EPSILON_ANG
    pRK45 = pRK45_t(tol, h_max, h_min)
    cs = cs_normal_(elli, beta1, L1, beta2, L2)
    for i in range(1,30):
        alpha, dist, alpha_beta, dist_beta, alpha_L, dist_L = partials_inverse_dist_normal_(pRK45, plat, elli, beta1, L1, beta2, L2)
        a = alpha_beta
        b = alpha_L
        c = dist_beta
        d = dist_L        
        F1 = alpha - ALPHA
        F2 = dist - DIST
        det = a*d - b*c
        sgndet = sgn(det)
        deltaBeta = ( d*F1 - b*F2)/(det + sgndet*EPSILON_MAQ)
        deltaL    = (-c*F1 + a*F2)/(det + sgndet*EPSILON_MAQ)
        if abs(deltaL) > MAXSTEP_ANG:
            deltaL = math.copysign(MAXSTEP_ANG, deltaL)
        if abs(deltaBeta) > MAXSTEP_ANG:
            deltaBeta = math.copysign(MAXSTEP_ANG, deltaBeta)
        beta2 = beta2 - deltaBeta
        L2 = L2 - deltaL
        cs = cs_normal_(elli, beta1, L1, beta2, L2)
        if max(math.fabs(deltaBeta), math.fabs(deltaL)) < EPSILON_ANG:
            break
    sbeta2, dummy = sincos(beta2)
    beta2 = math.asin(sbeta2)
    sL2, cL2 = sincos(L2)
    L2 = arg(cL2, sL2)
    return beta2phi(elli, beta2), L2

def inv_normal_dist(plat, elli, phi1:float, L1:float, phi2:float, L2:float, L11:float, L22:float):
    alphas = []
    tol = EPSILON_ANG; h_max = MAXSTEPRK45_ANG; h_min = EPSILON_ANG    
    pRK45 = pRK45_t(tol, h_max, h_min)
    beta1 = phi2beta(elli, phi1)
    beta2 = phi2beta(elli, phi2)
    cs = cs_normal_(elli, beta1, L1, beta2, L2)
    beta11 = L2beta_normal(elli, cs, beta1, L11)
    beta22 = L2beta_normal(elli, cs, beta2, L22)
    Beta0, L0 = CalcBeta0_normal(elli, cs, beta2, 0.5*(L2-L1))
    alpha, dist, lambdas, betas, dists = inverse_dist_normal_(pRK45, plat, elli, cs, beta11, L11, beta22, L22)
    phis = []
    Phi0 = beta2phi(elli, Beta0)
    for i in range(0, len(betas)):
        phis.append(RAD2DEG(beta2phi(elli, betas[i])))
        alphas.append(RAD2DEG(acimut_normal_(elli, cs, betas[i], lambdas[i])))
        lambdas[i] = RAD2DEG(lambdas[i])

    pathpoints = np.column_stack((phis, lambdas, alphas, dists))
    return alpha, dist, Phi0, L0, pathpoints

def inv_normal_area(plat, elli, phi1:float, L1:float, phi2:float, L2:float, L11:float, L22:float):
    alphas = []
    tol = EPSILON_ANG; h_max = MAXSTEPRK45_ANG; h_min = EPSILON_ANG
    pRK45 = pRK45_t(tol, h_max, h_min)
    beta1 = phi2beta(elli, phi1)
    beta2 = phi2beta(elli, phi2)
    cs = cs_normal_(elli, beta1, L1, beta2, L2)
    beta11 = L2beta_normal(elli, cs, beta1, L11)
    beta22 = L2beta_normal(elli, cs, beta2, L22)
    Beta0, L0 = CalcBeta0_normal(elli, cs, beta2, 0.5*(L2-L1))
    alpha, dist, area, lambdas, betas, dists, areas = inverse_area_normal_(pRK45, plat, elli, cs, beta11, L11, beta22, L22)
    phis = []
    Phi0 = beta2phi(elli, Beta0)
    for i in range(0, len(betas)):
        phis.append(RAD2DEG(beta2phi(elli, betas[i])))
        alphas.append(RAD2DEG(acimut_normal_(elli, cs, betas[i], lambdas[i])))
        lambdas[i] = RAD2DEG(lambdas[i])

    pathpoints = np.column_stack((phis, lambdas, alphas, dists, areas))
    return alpha, dist, area, Phi0, L0, pathpoints

def beta_normal(plat, elli, phi1:float, L1:float, phi2:float, L2:float, L):
    alphas = []
    tol = EPSILON_ANG; h_max = MAXSTEPRK45_ANG; h_min = EPSILON_ANG    
    pRK45 = pRK45_t(tol, h_max, h_min)
    beta1 = phi2beta(elli, phi1)
    beta2 = phi2beta(elli, phi2)
    cs = cs_normal_(elli, beta1, L1, beta2, L2)
    sL, cL = sincos(L)
    D = cs[1]*cL + cs[2]*sL
    beta = math.atan(cs[3]/D) + math.acos(-cs[4]/math.hypot(D, cs[3]))
    print(sgn(D))
    beta = math.asin(math.sin(beta))
    return beta

def CalcBeta0_normal(elli, cs, beta, L):
    sbeta, cbeta = sincos(beta)
    sL, cL = sincos(L)
    for _ in range(30):
        g_beta = -cs[1]*sbeta*cL - cs[2]*sbeta*sL + cs[3]*cbeta
        g_L = -cs[1]*cbeta*sL + cs[2]*cbeta*cL
        g = cs[1]*cbeta*cL + cs[2]*cbeta*sL + cs[3]*sbeta + cs[4]
        g_L_B = cs[1]*sbeta*sL - cs[2]*sbeta*cL
        g_L_L = -cs[1]*cbeta*cL - cs[2]*cbeta*sL
        a = g_beta
        b = g_L
        c = g_L_B
        d = g_L_L
        F1 = g
        F2 = g_L
        det = a*d - b*c
        sgndet = sgn(det)
        deltaBeta = ( d*F1 - b*F2)/(det + sgndet*EPSILON_MAQ)
        deltaL    = (-c*F1 + a*F2)/(det + sgndet*EPSILON_MAQ)
        if abs(deltaL) > MAXSTEP_ANG:
            deltaL = math.copysign(MAXSTEP_ANG, deltaL)

        if abs(deltaBeta) > MAXSTEP_ANG:
            deltaBeta = math.copysign(MAXSTEP_ANG, deltaBeta)

        beta = beta - deltaBeta        
        L = L - deltaL
        sbeta, cbeta = sincos(beta)
        sL, cL = sincos(L)
        if max(math.fabs(deltaBeta), math.fabs(deltaL)) < EPSILON_ANG:
            break

    beta = math.asin(sbeta)
    L = arg(cL, sL)
    return beta, L

#-------------------------------------------------------------------------------------------------
import simplekml
from shapely.geometry import Point, LineString
import geopandas as gpd
import pandas as pd

def guardar_curva_kmz(pathpoints, nombre_archivo):
    kml = simplekml.Kml()
    ls = kml.newlinestring(name="Curva Área Central Sect")        
    coords = [(lon, lat) for lat, lon in pathpoints[:, :2]]
    ls.coords = coords
    ls.style.linestyle.color = simplekml.Color.red  
    ls.style.linestyle.width = 3    
    html_tabla = """
    <table border="1" cellpadding="3" style="border-collapse: collapse; font-family: monospace;">
      <tr style="background-color: #f2f2f2;">
        <th>Lat</th><th>Lon</th><th>Acimut</th><th>Dist</th><th>Area</th>
      </tr>
    """
    for fila in pathpoints:
        lat, lon, azim, dist, area = fila
        html_tabla += f"<tr><td>{lat:.6f}</td><td>{lon:.6f}</td><td>{azim:.4f}</td><td>{dist:.4f}</td><td>{area:.4f}</td></tr>\n"
    
    html_tabla += "</table>"
    ls.description = html_tabla
    kml.savekmz(nombre_archivo)

def guardar_como_point_shapefile(pathpoints, nombre_archivo):
    df = pd.DataFrame(pathpoints, columns=['Latitud', 'Longitud', 'Acimut', 'Distancia', 'Area'])    
    geometria = [Point(xy) for xy in zip(df['Longitud'], df['Latitud'])]    
    gdf = gpd.GeoDataFrame(df, geometry=geometria, crs="EPSG:4326")
    gdf.to_file(nombre_archivo)
    print(f"Shapefile guardado como '{nombre_archivo}' con {len(gdf)} puntos y 5 columnas de datos.")

def guardar_como_linea_shapefile(pathpoints, nombre_archivo="curva_linea.shp"):
    coords = [(lon, lat) for lat, lon in pathpoints[:, :2]]
    linea = LineString(coords)
    gdf_linea = gpd.GeoDataFrame(geometry=[linea], crs="EPSG:4326")
    gdf_linea.to_file(nombre_archivo)

#-------------------------------------------------------------------------------------------------

import argparse
import os
import csv
import numpy as np

# Asumo que estas funciones y constantes ya están definidas en tu archivo original
# from tu_modulo import (GRS80_a, MAXSTEPRK45_ANG_INITIALVALUE, RAD2DEG, DEG2RAD, 
#                         pelipsoide_init, init_platn6, inv_align_area, inv_normal_area, 
#                         inv_central_area, direct_curva_align_, direct_curva_normal_, 
#                         direct_curva_central_, guardar_curva_kmz, guardar_como_point_shapefile, 
#                         guardar_como_linea_shapefile)

import re

def leer_poligono_archivo(filepath):
    """
    Lee un archivo de texto/CSV con coordenadas de vértices de un polígono.
    - Acepta separación por coma (,) y/o tabulación (\t).
    - Ignora la primera línea si es una cabecera (ej. "lat,lon,h" o "id\ty\tx").
    - Ignora alturas y etiquetas.
    Formatos permitidos por línea:
      - etiqueta, lat, lon, [altura]
      - lat, lon, [altura]
    """
    vertices = []
    
    def extraer_coordenadas(parts):
        """Intenta extraer lat, lon de una lista de strings. Retorna (lat, lon) o None."""
        lat, lon = None, None
        try:
            if len(parts) == 2:
                # Formato: lat, lon
                lat, lon = float(parts[0]), float(parts[1])
            elif len(parts) == 3:
                # Podría ser: etiqueta, lat, lon  O  lat, lon, altura
                try:
                    lat, lon = float(parts[0]), float(parts[1])
                except ValueError:
                    lat, lon = float(parts[1]), float(parts[2])
            elif len(parts) >= 4:
                # Formato: etiqueta, lat, lon, altura (se ignoran columnas extra)
                lat, lon = float(parts[1]), float(parts[2])
            else:
                return None
        except ValueError:
            # Si falla la conversión a float, no son coordenadas válidas
            return None
            
        return (lat, lon)

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            # Separar por comas y/o tabulaciones (1 o más ocurrencias)
            parts = re.split(r'[,\t]+', line)
            
            # Limpiar espacios y comillas
            parts = [p.strip().strip('"').strip("'") for p in parts]
            parts = [p for p in parts if p]  # Eliminar vacíos
            
            resultado = extraer_coordenadas(parts)
            
            if resultado is not None:
                vertices.append(resultado)
            else:
                # Si la línea no se pudo parsear y es la primera, asuminos que es cabecera
                if line_num == 1:
                    continue
                else:
                    print(f"Advertencia: Línea {line_num} ignorada por formato incorrecto: {line}")
                    
    return vertices

def main():
    global MAXSTEPRK45_ANG
    MAXSTEPRK45_ANG = MAXSTEPRK45_ANG_INITIALVALUE
    pathpoints = None
    csv_pathpoints = None
    resultados = {}
    parser = argparse.ArgumentParser(
        description="Cálculo de problemas directos e inversos de curvas de alineacción, sección normal y sección central",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Grupo mutuamente excluyente: solo se puede -i, -d o -poly
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-i', '--inverso', action='store_true', help='Ejecutar problema inverso')
    group.add_argument('-d', '--directo', action='store_true', help='Ejecutar problema directo')
    group.add_argument('-poly', '--poly-sup', type=str, metavar='ARCHIVO', 
                        help='Calcula la superficie dentro de un polígono dado en un archivo CSV/TXT')

    # Parámetros generales
    parser.add_argument('-t', '--tipo', choices=['align', 'normal', 'central'], default='central',
                        help='Tipo de sección geodésica (defecto: central)')
    parser.add_argument('-P1', nargs=2, type=float, metavar=('LAT', 'LON'), required=False,
                        help='Punto 1: latitud longitud (en grados decimales). Requerido para -i y -d')
    parser.add_argument('-e', nargs=2, type=float, metavar=('A', 'INV_F'), default=[GRS80_a, 298.2572221008827],
                        help='Elipsoide: semieje_mayor inversa_aplastamiento (defecto: GRS80)')
    parser.add_argument('-o', '--output', type=str, metavar='NOMBRE_ARCHIVO',
                        help='Nombre base para guardar salidas (KMZ, SHP, CSV)')

    # Parámetros exclusivos de cada problema
    parser.add_argument('-P2', nargs=2, type=float, metavar=('LAT', 'LON'),
                        help='Punto 2: latitud longitud (en grados decimales). Requerido para -i')
    parser.add_argument('-az', type=float, metavar='GRADOS',
                        help='Acimut inicial (en grados decimales). Requerido para -d')
    parser.add_argument('-s', type=float, metavar='METROS',
                        help='Distancia (en metros). Requerido para -d')
    parser.add_argument('-mstep', '--max-step', type=float, metavar='GRADOS', default = RAD2DEG(MAXSTEPRK45_ANG))

    args = parser.parse_args()

    # --- Paso máximo en grados decimales (a radianes) ---
    MAXSTEPRK45_ANG = DEG2RAD(args.max_step)

    # --- Validaciones ---
    if args.inverso and not args.P2:
        parser.error("El problema inverso (-i) requiere el parámetro -P2")
    if args.directo and (args.az is None or args.s is None):
        parser.error("El problema directo (-d) requiere los parámetros -az y -s")
    if (args.inverso or args.directo) and not args.P1:
        parser.error("Los problemas directo (-d) e inverso (-i) requieren el parámetro -P1")

    # --- Inicialización del Elipsoide ---
    a_elli, inv_f_elli = args.e
    f_elli = 1.0 / inv_f_elli if inv_f_elli != 0 else 0.0
    elli = pelipsoide_init(f_elli, a_elli)
    plat = init_platn6(elli.n)

    pathpoints = None
    resultados = {}

    # =====================================================================
    # PROBLEMA INVERSO
    # =====================================================================
    if args.inverso:
        phi1 = DEG2RAD(args.P1[0])
        L1 = DEG2RAD(args.P1[1])
        phi2 = DEG2RAD(args.P2[0])
        L2 = DEG2RAD(args.P2[1])

        if args.tipo == 'align':
            alpha, dist, area, Phi0, L0, pathpoints = inv_align_area(plat, elli, phi1, L1, phi2, L2, L1, L2)
        elif args.tipo == 'normal':
            alpha, dist, area, Phi0, L0, pathpoints = inv_normal_area(plat, elli, phi1, L1, phi2, L2, L1, L2)
        elif args.tipo == 'central':
            alpha, dist, area, Phi0, L0, pathpoints = inv_central_area(plat, elli, phi1, L1, phi2, L2, L1, L2)
        
        csv_pathpoints = pathpoints.copy()
        
        # --- Imprimir Resultados en Consola ---
        print("\n" + "="*40)
        print(f"Acimut (deg): {RAD2DEG(alpha):.10f}")
        print(f"Distancia (m): {dist:.4f}")
        print(f"Area (m2): {area:.0f}")
        print(f"Latitud Vértice phi0 (deg): {Phi0:.10f}")
        print(f"Longitud Vértice L0 (deg): {L0:.10f}")
        print("="*40 + "\n")

    # =====================================================================
    # PROBLEMA DIRECTO
    # =====================================================================
    elif args.directo:
        phi1 = DEG2RAD(args.P1[0])
        L1 = DEG2RAD(args.P1[1])
        alpha = DEG2RAD(args.az)
        dist = args.s

        if args.tipo == 'align':
            phi2, L2 = direct_curva_align_(plat, elli, phi1, L1, alpha, dist)
        elif args.tipo == 'normal':
            phi2, L2 = direct_curva_normal_(plat, elli, phi1, L1, alpha, dist)
        elif args.tipo == 'central':
            phi2, L2 = direct_curva_central_(plat, elli, phi1, L1, alpha, dist)

        
        
        # --- Imprimir Resultados en Consola ---
        print("\n" + "="*40)
        print(f"Latitud P2 (deg): {RAD2DEG(phi2):.10f}")
        print(f"Longitud P2 (deg): {RAD2DEG(L2):.10f}")        
        print("="*40 + "\n")

        # Para poder exportar la curva en el problema directo, 
        # necesitamos los pathpoints, por lo que calculamos el inverso hacia el P2 encontrado.
        if args.output:
            if args.tipo == 'align':
                _, _, _, _, _, pathpoints = inv_align_area(plat, elli, phi1, L1, phi2, L2, L1, L2)
            elif args.tipo == 'normal':
                _, _, _, _, _, pathpoints = inv_normal_area(plat, elli, phi1, L1, phi2, L2, L1, L2)
            elif args.tipo == 'central':
                _, _, _, _, _, pathpoints = inv_central_area(plat, elli, phi1, L1, phi2, L2, L1, L2)
            csv_pathpoints = pathpoints.copy()

        # =====================================================================
    # SUPERFICIE DE POLÍGONO
    # =====================================================================
    elif args.poly_sup:
        vertices_grados = leer_poligono_archivo(args.poly_sup)
        
        if len(vertices_grados) < 3:
            print("Error: Se requieren al menos 3 vértices para formar un polígono.")
            return
        
        # Convertir a radianes
        vertices_rad = [(DEG2RAD(lat), DEG2RAD(lon)) for lat, lon in vertices_grados]
        
        # Filtrar vértices duplicados consecutivos 
        vertices_filtrados = [vertices_rad[0]]
        for v in vertices_rad[1:]:
            if v != vertices_filtrados[-1]:
                vertices_filtrados.append(v)
        vertices_rad = vertices_filtrados
        
        if len(vertices_rad) < 3:
            print("Error: Se requieren al menos 3 vértices únicos no consecutivos para formar un polígono.")
            return
        
        total_area = 0.0
        segments = []  # Arreglos numpy para KMZ/SHP (con puntos intermedios)
        propiedades_aristas = [] # (distancia, acimut, area) del tramo saliente de cada vértice
        
        print("\n" + "="*50)
        print(f"Calculando polígono con {len(vertices_rad)} vértices...")
        
        # Iterar sobre las aristas del polígono (P_i -> P_i+1)
        for i in range(len(vertices_rad)):
            p1 = vertices_rad[i]
            p2 = vertices_rad[(i + 1) % len(vertices_rad)] # Conecta el último con el primero
            
            # Capturamos alpha, dist y area
            if args.tipo == 'align':
                alpha, dist, area, _, _, path_seg = inv_align_area(plat, elli, p1[0], p1[1], p2[0], p2[1], p1[1], p2[1])
            elif args.tipo == 'normal':
                alpha, dist, area, _, _, path_seg = inv_normal_area(plat, elli, p1[0], p1[1], p2[0], p2[1], p1[1], p2[1])
            elif args.tipo == 'central':
                alpha, dist, area, _, _, path_seg = inv_central_area(plat, elli, p1[0], p1[1], p2[0], p2[1], p1[1], p2[1])
            
            total_area += area
            segments.append(path_seg)
            propiedades_aristas.append((dist, RAD2DEG(alpha), area))
            
        # Concatenar los segmentos en un solo arreglo numpy 2D (para KMZ y SHP)
        if segments:
            full_pathpoints = segments[0]
            for seg in segments[1:]:
                full_pathpoints = np.vstack((full_pathpoints, seg[1:]))
            full_pathpoints[-1] = full_pathpoints[0]
            pathpoints = full_pathpoints
        else:
            pathpoints = None

        # =====================================================================
        # CONSTRUIR ARREGLO LIMPIO PARA EL CSV (Lat, Lon, Acimut, Dist, Area)
        # =====================================================================
        csv_rows = []
        for i in range(len(vertices_rad)):
            lat_deg = RAD2DEG(vertices_rad[i][0])
            lon_deg = RAD2DEG(vertices_rad[i][1])
            dist_out, azi_out, area_out = propiedades_aristas[i]
            csv_rows.append([lat_deg, lon_deg, azi_out, dist_out, area_out])
        
        # Cerrar el polígono repitiendo el primer vértice con sus mismos datos
        #csv_rows.append(csv_rows[0])
        csv_pathpoints = np.array(csv_rows)
            
        # --- Imprimir el resumen en consola ---
        print("-"*50)
        print(f"{'Arista':<10} {'Distancia (m)':<15} {'Acimut (deg)':<15}")
        print("-"*50)
        for i in range(len(vertices_rad)):
            idx_from = i + 1
            idx_to = (i + 1) % len(vertices_rad) + 1
            d, a, _ = propiedades_aristas[i]
            print(f"{idx_from} - {idx_to:<5} {d:<15.4f} {a:<15.8f}")
        print("-"*50)
        print(f"Superficie del polígono (m2): {abs(total_area):.2f}")
        print("="*50 + "\n")

    # --- Exportar Archivos si existe (-o) ---
    if args.output and pathpoints is not None:
        base_name = args.output
        dir_name = os.path.dirname(base_name)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)

        print(f"Generando archivos: {base_name}.* ...")
        try:
            # KMZ y SHP usan el arreglo con todos los puntos interpolados (curvas suaves)
            guardar_curva_kmz(pathpoints, f"{base_name}.kmz")
            guardar_como_point_shapefile(pathpoints, f"{base_name}_puntos.shp")
            guardar_como_linea_shapefile(pathpoints, f"{base_name}_linea.shp")
            
            # CSV usa el arreglo limpio, sin puntos intermedios y con datos del tramo saliente
            np.savetxt(f"{base_name}.csv", csv_pathpoints, delimiter=",", fmt='%.10f')
            print("Archivos generados\n")
        except Exception as e:
            print(f"Error al guardar los archivos: {e}\n")
    elif args.output and pathpoints is None:
        print("No se generaron archivos, no hay puntos de la curva en pathpoints")

if __name__ == '__main__':
    main()