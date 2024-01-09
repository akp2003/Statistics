#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <utility>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <exception>
#include <map>
#include <string>
using namespace std;
const string VERSION="1.0";
//An Interface for probability distributions
class IDist
{
	virtual double _f(double x) = 0;

  public:
	virtual bool isInSx(double x) = 0;
	double f(double x)
	{
		if (isInSx(x))
			return _f(x);
		else
			return 0;
	}
	virtual double F(double x) = 0;
	double invF(double p, long prec = 100, long long max = 100000)
	{
		assert(p >= 0 && p <= 1);
		double x = 0;
		for (int i = 1; i <= prec; i++)
		{
			if (F(x) == p)
				return x;
			if (F(x) > p)
				x -= max / pow(2, i);
			else
				x += max / pow(2, i);
		}
		return x;
	}
};

class ICDist : public IDist
{
};

class NormDist : public ICDist
{
	double _mean;
	double _var;
	double _f(double x)
	{
		return exp(-0.5 * pow(x - _mean, 2) / _var) / sqrt(2 * M_PI * _var);
	}

  public:
	NormDist(double mean = 0, double var = 1) : _mean(mean), _var(var)
	{
		assert(var > 0);
	}
	bool isInSx(double x) { return true; }
	double F(double x)
	{
		return 0.5 * (1 + erf((x - _mean) / sqrt(_var * 2)));
	}
};

class TDist : public ICDist
{
	unsigned int _df;
	double _f(double x)
	{
		double v = _df;
		return tgamma((v + 1) / 2) * pow(1 + pow(x, 2) / v, -(v + 1) / 2) / (sqrt(M_PI * v) * tgamma(v / 2));
	}
	//https://cplusplus.com/forum/general/255896/
	double _hypergeometric(double a, double b, double c, double x)
	{
		const double TOLERANCE = 1.0e-10;
		double term = a * b * x / c;
		double value = 1.0 + term;
		int n = 1;

		while (abs(term) > TOLERANCE)
		{
			a++, b++, c++, n++;
			term *= a * b * x / c / n;
			value += term;
		}

		return value;
	}

  public:
	TDist(int df = 1) : _df(df)
	{
		assert(df > 0);
	}
	bool isInSx(double x) { return true; }
	double F(double x)
	{
		//double v = _df;
		boost::math::students_t dist(_df);
		return cdf(dist, x);
		//return 0.5 + (x * tgamma((v + 1) / 2)) * _hypergeometric(0.5, (v + 1) / 2, 1.5, -x * x / v) / (sqrt(M_PI * v) * tgamma(v / 2));
	}
};

class ChiSquareDist : public ICDist
{
	unsigned int _df;
	double _f(double x)
	{
		double v = _df;
		return (pow(x, (v / 2) - 1) * exp(-x / 2)) / (pow(2, v / 2) * tgamma(v / 2));
	}

  public:
	ChiSquareDist(int df = 1)
	{
		setDf(df);
	}

	bool isInSx(double x)
	{
		if (_df == 1 && x > 0)
			return true;
		if (_df != 1 && x >= 0)
			return true;
		return false;
	}

	double F(double x)
	{
		double v = _df;
		return boost::math::gamma_p(v / 2, x / 2);
	}
	void setDf(int df)
	{
		assert(df > 0);
		_df = df;
	}
	int getDf() const
	{
		return _df;
	}
	ChiSquareDist operator-(int num)
	{
		return ChiSquareDist(this->getDf() - num);
	}
};

template <int size>
class Sample
{
	double _mean;
	double _pmean;
	double _var;
	double _pvar;
	double *_data = NULL;
	bool isPMeanGiven = false;
	bool isPVarGiven = false;
	bool isVarGiven = false;
	bool _isDataGiven = false;

	double _calcVar(double n, double mean)
	{
		double x = 0;
		for (int i = 0; i < size; i++)
			x += pow(_data[i] - mean, 2);
		return x / n;
	}
	double _getPValue(double p, char side = '!')
	{
		switch (side)
		{
		case '!':
			return 2 * min(p, 1 - p);
		case '<':
			return p;
		case '>':
			return 1-p;
		}
		throw "invalid side";
	}

  public:
	bool isNormal = true;

	Sample(double mean) : _mean(mean)
	{
		_data = NULL;
	}
	Sample(double *data)
	{
		_data = new double[size];
		for (int i = 0; i < size; i++)
			_data[i] = data[i];
		_isDataGiven = true;

		//calculate mean
		double sum = 0;
		for (int i = 0; i < size; i++)
			sum += _data[i];
		_mean = sum / size;

		//calculate sample variance
		_var = _calcVar(size - 1, _mean);
		isVarGiven = true;
	}
	~Sample()
	{
		if (_data != NULL)
			delete (_data);
		//cout << "destructor called!";
	}
	// getter setter
	double getMean() const
	{
		return _mean;
	}
	double getPMean() const
	{
		if (!isPMeanGiven)
			throw "Population Mean is not Given";
		return _pmean;
	}
	double getPVar() const
	{
		if (!isPVarGiven)
			throw "Population Variance is not Given";
		return _pvar;
	}
	double getVar() const
	{
		if (!isVarGiven)
			throw "Sample Variance is not Given";
		return _var;
	}
	double getStd() const
	{
		return sqrt(getVar());
	}
	double getPStd() const
	{
		assert(isPVarGiven);
		return sqrt(_pvar);
	}
	int getSize() const
	{
		return size;
	}
	double getData(int index) const
	{
		assert(isDataGiven() && 0 <= index && index < size);
		return _data[index];
	}
	bool isDataGiven() const
	{
		return _isDataGiven;
	}

	void setVar(double var)
	{
		assert(var > 0 && !isDataGiven());
		isVarGiven = true;
		_var = var;
	}
	void setPVar(double pvar)
	{
		assert(pvar > 0);
		isPVarGiven = true;
		_pvar = pvar;
	}
	void setPMean(double pmean)
	{
		isPMeanGiven = true;
		_pmean = pmean;
	}

	//operator overloading
	double operator[](int index) const
	{
		return this->getData(index);
	}
	Sample operator-(const Sample &s)
	{
		if (size != s.getSize())
			throw "samples are not the same size";

		double _diff[s.getSize()];
		for (int i = 0; i < size; i++)
			_diff[i] = this->getData(i) - s[i];
		return Sample<size>(_diff);
	}
	template <int nsize>
	friend ostream &operator<<(ostream &out, const Sample<nsize> &s);

	//confidence interval
	pair<double, double> meanCI(double a = 0.05)
	{
		double d;
		TDist t(size - 1);
		NormDist n;
		if (isNormal)
			if (isPVarGiven)
				d = n.invF(1 - (a / 2)) * getPStd() / sqrt(size);
			else
				d = t.invF(1 - (a / 2)) * getStd() / sqrt(size);
		else
			throw "not implemented yet!";
		return make_pair(getMean() - d, getMean() + d);
	}
	pair<double, double> proportionCI(double a = 0.05)
	{
		assert(getMean() >= 0 && 1 >= getMean());
		double d;
		NormDist n;
		if (isNormal)
			d = n.invF(1 - (a / 2)) * sqrt(getMean() * (1 - getMean()) / size);
		else
			throw "not implemented yet!";
		return make_pair(getMean() - d, getMean() + d);
	}
	pair<double, double> varCI(double a = 0.05)
	{
		ChiSquareDist csd(size);
		if (isNormal)
			if (isPMeanGiven) //needs change : make a new variable for population estimation
				throw "not implemented yet!";
			else
				return make_pair((size - 1) * getVar() / (csd - 1).invF(1 - a/2), (size - 1) * getVar() / (csd - 1).invF(a/2));
		else
			throw "not implemented yet!";
	}
	template <int n1,int n2>
	friend pair<double, double> two_sample_meanCI(Sample<n1> s1,Sample<n2> s2,double a);
	template <int n1,int n2>
	friend pair<double, double> two_sample_proportionCI(Sample<n1> s1,Sample<n2> s2,double a);

	//hypothesis testing

	double TTest(double hypmean, char side = '!', double a = 0.05)
	{

		double t = (getMean()-hypmean) / (getStd() / sqrt(size));
		TDist td(size - 1);
		return _getPValue(td.F(t), side);
	}
	double ProportionTest(double hypp, char side = '!', double a = 0.05)
	{
		double z = (getMean()-hypp) / sqrt(getMean() * (1 - getMean()) / size);
		NormDist sn;
		return _getPValue(sn.F(z), side);
	}
	double VarTest(double hypvar, char side = '!', double a = 0.05)
	{
		//this function is incomplete!
		assert(hypvar > 0);
		double cs = (size - 1) * getVar() / hypvar;
		ChiSquareDist csd(size - 1);
		return _getPValue(csd.F(cs), side);
	}

	template <int n1,int n2>
	friend double two_sample_meanTest(Sample<n1> s1,Sample<n2> s2,double hypdelta,char side,double a);
	template <int n1,int n2>
	friend double two_sample_proportionTest(Sample<n1> s1,Sample<n2> s2,char side,double a);

};
//2 Sample confidence intervals
//confidence interval
//Note: I defined these functions friend because I may need to access private fields in the near future


template <int n1,int n2>
pair<double, double> two_sample_meanCI(Sample<n1> s1,Sample<n2> s2,double a = 0.05)
{
	//only not equal sigmas
	double d;
	TDist t(pow(s1.getVar()/s1.getSize() + s2.getVar()/s2.getSize(),2)/( pow(s1.getVar()/s1.getSize(),2)/(s1.getSize()-1) + pow(s2.getVar()/s2.getSize(),2)/(s2.getSize()-1)));
	double s=sqrt(s1.getVar()/s1.getSize() + s2.getVar()/s2.getSize());
	if (s1.isNormal && s2.isNormal)
		d = t.invF(1 - (a / 2)) *s;
	else
		throw "not implemented yet!";
	double deltamean=s1.getMean()-s2.getMean();
	return make_pair(deltamean - d, deltamean + d);
}
template <int n1,int n2>
pair<double, double> two_sample_proportionCI(Sample<n1> s1,Sample<n2> s2,double a = 0.05)
{
	//only not equal sigmas
	double d;
	NormDist n;
	double rp=sqrt((s1.getMean()*(1-s1.getMean()))/s1.getSize() + (s2.getMean()*(1-s2.getMean()))/s2.getSize());
	if (s1.isNormal && s2.isNormal)
		d = n.invF(1 - (a / 2)) *rp;
	else
		throw "not implemented yet!";
	double deltamean=s1.getMean()-s2.getMean();
	return make_pair(deltamean - d, deltamean + d);
}

template <int n1,int n2>
double two_sample_meanTest(Sample<n1> s1,Sample<n2> s2,double hypdelta=0,char side = '!',double a=0.05)
{
	double s=sqrt(s1.getVar()/s1.getSize() + s2.getVar()/s2.getSize());
	double deltamean=s1.getMean()-s2.getMean();
	double t = (deltamean- hypdelta) / s;
	TDist td(pow(s1.getVar()/s1.getSize() + s2.getVar()/s2.getSize(),2)/( pow(s1.getVar()/s1.getSize(),2)/(s1.getSize()-1) + pow(s2.getVar()/s2.getSize(),2)/(s2.getSize()-1)));
	return s1._getPValue(td.F(t), side);
}

template <int n1,int n2>
double two_sample_proportionTest(Sample<n1> s1,Sample<n2> s2,char side = '!',double a=0.05)
{
	double p=(n1*s1.getMean()+n2*s2.getMean())/(n1+n2);
	double rp=sqrt((p*(1-p))/s1.getSize() + (p*(1-p))/s2.getSize());
	double deltamean=s1.getMean()-s2.getMean();
	double z = (deltamean) / rp;
	NormDist n;
	return s1._getPValue(n.F(z), side);
}


template <int size>
ostream &operator<<(ostream &out, const Sample<size> &samp)
{
	//hardest part of this project!!!!
	//ethical lessons:
	//dont use bind becuase it call destructor!!!!
	//it is so hard to use pointer to member function but i believe it is worth it!!!
	typedef double (Sample<size>::*func)() const;
	//https://stackoverflow.com/questions/7582546/using-generic-stdfunction-objects-with-member-functions-in-one-class
	map<string, func> list = {
		{"Sample Mean", &Sample<size>::getMean},
		{"Sample Variance", &Sample<size>::getVar},
		{"Sample STD", &Sample<size>::getStd},
		{"Population Mean", &Sample<size>::getPMean},
		{"Population Variance", &Sample<size>::getPVar},
	};
	//(samp.*list["Sample Mean"])();
	typename map<string, func>::iterator it = list.begin();
	while (it != list.end())
	{
		try
		{
			double val = (samp.*(it->second))();
			out << it->first << ":" << val << endl;
		}
		catch (const char *c)
		{
			//[optional] you can show error but it gets messy!
			//out << c << endl;
		}
		++it;
	}
	return out;
}

//Unit Test :
//https://stackoverflow.com/questions/52273110/how-do-i-write-a-unit-test-in-c
#define IS_TRUE(x)                                                                    \
	{                                                                                 \
		if (!(x))                                                                     \
			std::cout << __FUNCTION__ << " failed on line " << __LINE__ << std::endl; \
	}

bool isSame(double a, double b, double e = 0.0000001)
{
	return fabs(a - b) < e;
}

void unit_test()
{
	TDist t;
	TDist t8(8);
	ChiSquareDist cs(6);
	//cout << cs.F(6)<<endl;
	IS_TRUE(isSame(t.f(0), 0.318, 0.001));

	IS_TRUE(isSame(t8.f(0), 0.387, 0.001));
	IS_TRUE(isSame(t8.f(1.2), 0.1836, 0.001));
	IS_TRUE(isSame(t8.F(1.2), 0.86776644, 0.000001));
	IS_TRUE(isSame(t8.invF(0.867766), 1.2, 0.0001));

	IS_TRUE(isSame(cs.f(5), 0.1282578103, 0.00001));
	IS_TRUE(isSame(cs.F(5), 0.4561868841, 0.00001));
	IS_TRUE(isSame(cs.invF(0.95), 12.59158724, 0.00001));
	IS_TRUE(isSame((cs - 3).f(12), 0.0034255775, 0.0000001));
}

int main()
{
	unit_test();

	pair<double, double> mc;
	//Ex 1
	{
	cout << "Example 1:" << endl;
	double data[36] = {
		22.2, 23.9, 24.1, 21.7, 25.9, 18.4, 24.8, 28.2, 17.3, 26.4, 21.2, 29.3, 23.2, 21.9, 25.2, 26.4, 22.6, 24.7, 23.9, 30.8, 25.0, 19.1, 23.5, 28.8, 27.1, 20.4, 27.2, 23.5, 19.3, 24.7, 29.9, 21.3, 27.1, 26.6, 20.0, 25.8};
	Sample<36> s(data);
	cout << s;
	mc = s.meanCI();
	cout << mc.first << " " << mc.second << endl;
	cout << "p-value: " << s.TTest(25) << endl;
	pair<double, double> vc = s.varCI();
	cout << sqrt(vc.first) << " " << sqrt(vc.second) << endl;
	cout << s.VarTest(9) << endl;
	cout << endl;
	}
	//Ex 2
	{
	cout << "Example 2:" << endl;
	Sample<1000> s2(0.546);
	pair<double, double> pc = s2.proportionCI();
	cout << pc.first << " " << pc.second << endl;
	cout << "p-value: " << s2.ProportionTest(0.5, '>') << endl;
	cout << endl;
	}
	//Ex 3
	{
	cout << "Example 3:" << endl;
	double data1[10] = {25,7,3,7,15,25,12,6,15,10};
	double data2[10] = {18,1,12,20,21,33,38,40,44,48};
	Sample<10> s1(data1);
	Sample<10> s2(data2);
	mc=two_sample_meanCI(s1,s2);
	cout << s1 <<endl <<s2;
	cout << mc.first << " " << mc.second << endl;
	cout << two_sample_meanTest(s1,s2) << endl;
	cout << endl;
	}
	//Ex 4
	{
	cout << "Example 4:" << endl;
	Sample<1000> s1(0.546);
	Sample<1000> s2(0.475);
	pair<double, double> pc = two_sample_proportionCI(s1,s2);
	cout << pc.first << " " << pc.second << endl;
	cout << two_sample_proportionTest(s1,s2) << endl;
	cout << endl;
	}
	//Ex 11
	cout << "Example 11:" << endl;
	double before[5] = {84, 97, 78, 91, 85};
	double after[5] = {80, 98, 75, 90, 82};
	Sample<5> bs(before);
	Sample<5> as(after);
	Sample<5> diff = bs - as;
	cout << diff;
	mc = diff.meanCI();
	cout << mc.first << " " << mc.second << endl;
	cout << diff.TTest(0) << endl;
}

