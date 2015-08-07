/*
cppUtility.cpp
Utilities for non-gpu stuff.

Nicolas Sawaya
2013
*/


cuDoubleComplex get_neg_inv_hbar_imag() {


    cuDoubleComplex neg_inv_hbar_imag;
    double hbar_inv_cm_fs, lightspeed;



    //double planck_J_s = 6.62606957e-34; //J s
    double pi = atan(1)*4;
    //double hbar_J_s = planck_J_s/(2*pi); //J s
    lightspeed = 299792548.; //m/s
    hbar_inv_cm_fs = (1/100.) * (1e15) * 1./(lightspeed * 2*pi); //cm^-1 * fs

    cout << "hbar_inv_cm_fs = " << hbar_inv_cm_fs << endl;

    neg_inv_hbar_imag.x = 0;
    neg_inv_hbar_imag.y = -1./hbar_inv_cm_fs; // i*(1/hbar)

    cout << "neg_inv_hbar_imag.x = " << neg_inv_hbar_imag.x << endl;
    cout << "neg_inv_hbar_imag.y = " << neg_inv_hbar_imag.y << endl;


    return neg_inv_hbar_imag;

}

cuDoubleComplex powMult(cuDoubleComplex i, cuDoubleComplex j) {

    cuDoubleComplex returnVal;
    double a = i.x;
    double b = i.y;
    double c = j.x;
    double d = j.y;

    returnVal.x = a*c - b*d;
    returnVal.y = a*d + b*c;

    return returnVal;

}

cuDoubleComplex powComplex(cuDoubleComplex inVal, int n) {

    cuDoubleComplex outVal;
    outVal.x = 1;
    outVal.y = 0;

    if(n==0) {
        return outVal;
    }
    // if(n==1) {
    //     return inVal;
    // }

    for(int i=0;i<n;i++) {
        outVal = powMult(outVal, inVal);
    }

    return outVal;

}




int helperFactorial(int n)
{
    return (n == 1 || n == 0) ? 1 : helperFactorial(n - 1) * n;
}


void stringSplit(string& str, string& str1, string& str2, char chrSplit) {

    int pos = str.find(chrSplit);

    str2 = str.substr( pos+2, str.size()-1);
    str1 = str.substr(0,pos);
        
}


void stringTrim(string& str)
{
    //Remove tabs
    string::size_type pos1 = str.find_first_not_of('\t');
    string::size_type pos2 = str.find_last_not_of('\t');
    str = str.substr(pos1 == string::npos ? 0 : pos1,
        pos2 == string::npos ? str.length() - 1 : pos2 - pos1 + 1);


        //Remove spaces
        pos1 = str.find_first_not_of(' ');
        pos2 = str.find_last_not_of(' ');
        str = str.substr(pos1 == string::npos ? 0 : pos1,
                pos2 == string::npos ? str.length() - 1 : pos2 - pos1 + 1);
}





