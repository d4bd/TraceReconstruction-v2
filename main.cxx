#include "Rivelatore.h"
#include "Simulazione.h"
#include "Analysis.h"

int main(){

    Rivelatore detector;
 
    SimulatePoint("auto", detector, 100, 0.7,1, true, true);
    Analisys("auto", 0.005, 0.1, true, true, true, false, false);

	return 0;
}
