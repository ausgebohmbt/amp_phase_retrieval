//#include "extcode.h"
//#pragma pack(push)
//#pragma pack(1)

//#ifdef __cplusplus
//extern "C" {
//#endif

/*!
 * Open_Dev
 */
int32_t __stdcall Open_Dev(uint8_t bIDList[], int32_t bIDSize);
/*!
 * Close_Dev
 */
int32_t __stdcall Close_Dev(uint8_t bIDList[], int32_t bIDSize);
/*!
 * Check_DrivUniqSerial
 */
int32_t __stdcall Check_DrivUniqSerial(uint8_t bID, uint32_t *DevUniqSerial);
/*!
 * Check_HeadSerial
 */
int32_t __stdcall Check_HeadSerial(uint8_t bID, char HeadSerial[], 
	int32_t CharSize);
/*!
 * Write_FMemArray
 */
int32_t __stdcall Write_FMemArray(uint8_t bID, uint8_t ArrayIn[], 
	int32_t ArraySize, uint32_t XPixel, uint32_t YPixel, uint32_t SlotNo);
/*!
 * Write_FMemBMPPath
 */
int32_t __stdcall Write_FMemBMPPath(uint8_t bID, char Path[], 
	uint32_t SlotNo);
/*!
 * Chenge_DispSlot
 */
int32_t __stdcall Chenge_DispSlot(uint8_t bID, uint32_t SlotNo);
/*!
 * Check_Temp
 */
int32_t __stdcall Check_Temp(uint8_t bID, double *HeadTemp, double *CBTemp);

//MgErr __cdecl LVDLLStatus(char *errStr, int errStrLen, void *module);

//void __cdecl SetExcursionFreeExecutionSetting(Bool32 value);

//#ifdef __cplusplus
//} // extern "C"
//#endif

