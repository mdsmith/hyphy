/*	Button component for Win 32.		Sergei L. Kosakovsky Pond, May 2000-December 2002.*/#include "HYLabel.h"#include "HYPullDown.h"#include "errorfns.h"#include "HYPlatformWindow.h"#include "HYEventTypes.h"		//__________________________________________________________________static LRESULT	CALLBACK pullDownSubclassHandler (HWND WindowHand, UINT iMsg, WPARAM wParam, LPARAM lParam){	_HYPullDown * theParent = (_HYPullDown*)GetWindowLongPtr (WindowHand, GWLP_USERDATA);		switch (iMsg)	{		case WM_LBUTTONDOWN:		case WM_LBUTTONDBLCLK:		{			if (theParent->IsEnabled())			{				POINT loc = {theParent->menuRect.left, theParent->menuRect.top};				ClientToScreen (theParent->parentWindow, &loc);												if (theParent->messageRecipient)					theParent->messageRecipient->ProcessEvent (generateMenuOpenEvent (theParent->GetID()));				SetMenuDefaultItem (theParent->theMenu, theParent->selection, true);				long tSel = TrackPopupMenu (theParent->theMenu, TPM_LEFTALIGN|TPM_TOPALIGN|TPM_RETURNCMD|TPM_LEFTBUTTON|TPM_NONOTIFY, 													 loc.x, 													 loc.y, 													 0, theParent->parentWindow, NULL);				if (tSel)				{					theParent->selection = tSel-1;					theParent->SendSelectionChange();					theParent->_RefreshComboBox();				}									}			return 0L;		}	}		return CallWindowProc ((WNDPROC)theParent->mainHandler, WindowHand, iMsg, wParam, lParam);}//___________________________________________________________________HYPlatformPullDown::_HYPlatformPullDown(void){	myMenu   = CreateWindow ("COMBOBOX","",CBS_DROPDOWNLIST|WS_CHILD,0,0,100,25,((_HYPullDown*)this)->parentWindow,NULL,GetModuleHandle (NULL), NULL);	backFill = CreateSolidBrush(RGB(0xff,0xff,0xff));	theMenu  = CreatePopupMenu ();		if ((!backFill)||(!myMenu)||(!theMenu))	{		warnError (-108);	}	selection      = 0;	cbSelection    = -1;	menuWidth 	   = 100;		SetWindowLongPtr (myMenu,GWLP_USERDATA, (LONG_PTR)((_HYPullDown*)this));		mainHandler	= SetWindowLongPtr (myMenu,GWLP_WNDPROC,(LONG_PTR)pullDownSubclassHandler);}		//___________________________________________________________________HYPlatformPullDown::~_HYPlatformPullDown(void){	if (backFill)		DeleteObject (backFill);	if (theMenu)		DestroyMenu (theMenu);}//__________________________________________________________________void		_HYPlatformPullDown::_AdjustItemIDs  (long from, long to, long shift){	MENUITEMINFO     newItem;	newItem.cbSize = sizeof (MENUITEMINFO); 	newItem.fMask  = MIIM_ID;	for (long k=from; k<to; k++)	{		GetMenuItemInfo (theMenu, k, true, &newItem);		newItem.wID += shift;		SetMenuItemInfo (theMenu, k, true, &newItem);	}}//__________________________________________________________________void		_HYPlatformPullDown::_RefreshComboBox  (){	_HYPullDown * theParent = (_HYPullDown*)this;	if (myMenu&&((cbSelection!=selection)||(!theParent->IsEnabled())))	{			if (SendMessage (myMenu, CB_GETCOUNT, 0, 0))		{			SendMessage (myMenu, CB_DELETESTRING, 0, 0);		}				if (theParent->MenuItemCount())		{						if (selection<theParent->MenuItemCount())			{					_String * mItem = theParent->GetMenuItem (selection);								if (mItem->Equal(&menuSeparator))					mItem = &empty;										SendMessage (myMenu, CB_ADDSTRING, 0, (LPARAM)theParent->GetMenuItem (selection)->sData);				SendMessage (myMenu, CB_SETCURSEL, 0, 0);				cbSelection = selection;			}		}	}}//__________________________________________________________________void		_HYPlatformPullDown::_AddMenuItem  	(_String& newItemText, long index){		_String      newString (newItemText);		if (!myMenu) 		return;			MENUITEMINFO     newItem = {sizeof (MENUITEMINFO),0,0,0,0,nil,nil,nil,nil,0,nil};		_HYPullDown * theParent = (_HYPullDown*)this;		if (index<0)		index = theParent->MenuItemCount()-1;	if (newString == menuSeparator)	{		newItem.fMask = MIIM_TYPE;		newItem.fType = MFT_SEPARATOR;	}	else	{		newItem.fMask = MIIM_TYPE;		newItem.fType = MFT_STRING;				if (newString.sData[0] == '(')		{			newString.Trim (1,-1);			newItem.fState = MFS_GRAYED;		}				newItem.dwTypeData = newString.sData;						 					SIZE textSize;		HDC  theDC = GetDC(myMenu);	 	if(GetTextExtentPoint32(theDC,newString.sData,newString.sLength,&textSize))	 	{	 		if (textSize.cx+25>menuWidth)	 			menuWidth = textSize.cx+25;	 	}	 	ReleaseDC (myMenu, theDC);	}		newItem.fMask |= MIIM_ID|MIIM_STATE|MIIM_CHECKMARKS;	newItem.wID = index;		InsertMenuItem (theMenu, index, true, &newItem);	_AdjustItemIDs (index, theParent->MenuItemCount(),1);		if ((theParent->MenuItemCount()==1)||(selection==index))	{		cbSelection = -1;		_RefreshComboBox();	} }//__________________________________________________________________void		_HYPlatformPullDown::_SetMenuItem  	(_String& newItem, long index){		if (!myMenu) 		return;		if (index==selection)	{		cbSelection = -1;		_RefreshComboBox();	}		_DeleteMenuItem (index);	_AddMenuItem (newItem, index);}//__________________________________________________________________void		_HYPlatformPullDown::_MarkItem  	(long index, char mark){		if (!myMenu) 		return;		/*MENUITEMINFO     newItem;	newItem.cbSize = sizeof (MENUITEMINFO); 	newItem.fMask  = MIIM_STATE|MIIM_TYPE;	GetMenuItemInfo (theMenu, index, true, &newItem);	bool set = false;	if (mark)	{		if (!(newItem.fState & MFS_CHECKED))		{	 			newItem.fState |= MFS_CHECKED;			if (mark==HY_PULLDOWN_BULLET_MARK)				newItem.fType  |= MFT_RADIOCHECK;			else				if (newItem.fType & MFT_RADIOCHECK)					newItem.fType  -= MFT_RADIOCHECK;			set = true;		}	}	else		if (newItem.fState  & MFS_CHECKED)		{			newItem.fState -= MFS_CHECKED;			if (newItem.fType & MFT_RADIOCHECK)				newItem.fType  -= MFT_RADIOCHECK;			set = true;		}		if (set)		SetMenuItemInfo (theMenu, index, true, &newItem);*/	if (mark==HY_PULLDOWN_BULLET_MARK)		CheckMenuRadioItem (theMenu, index, index, index, MF_BYPOSITION);		else		CheckMenuItem (theMenu, index, MF_BYPOSITION|(mark?MF_CHECKED:MF_UNCHECKED));}//__________________________________________________________________char		_HYPlatformPullDown::_ItemMark  	(long index){		if (!myMenu) 		return 0;		MENUITEMINFO     newItem;	newItem.cbSize = sizeof (MENUITEMINFO); 	newItem.fMask  = MIIM_STATE|MIIM_TYPE;	GetMenuItemInfo (theMenu, index, true, &newItem);	if (newItem.fState & MFS_CHECKED)		return (newItem.fType&MFT_RADIOCHECK)?HY_PULLDOWN_BULLET_MARK:HY_PULLDOWN_CHECK_MARK;			return HY_PULLDOWN_NO_MARK;}//__________________________________________________________________void		_HYPlatformPullDown::_DeleteMenuItem  (long index){		if (!myMenu) 		return;		_HYPullDown * theParent = (_HYPullDown*)this;	DeleteMenu (theMenu, index, MF_BYPOSITION);			_AdjustItemIDs (index, theParent->MenuItemCount(),-1);	if (selection == index)	{		cbSelection = -1;		_RefreshComboBox ();	}}//__________________________________________________________________void		_HYPlatformPullDown::_SetBackColor (_HYColor& c){	if (backFill)			DeleteObject(backFill);			backFill = CreateSolidBrush(RGB(c.R,c.G,c.B));}//__________________________________________________________________long		_HYPlatformPullDown::_GetSelection (void){	return selection;}//__________________________________________________________________void		_HYPlatformPullDown::_Duplicate (Ptr p){	// unused}//__________________________________________________________________void		_HYPlatformPullDown::_Update (Ptr p){	_Paint (p);}//__________________________________________________________________void		_HYPlatformPullDown::_SetDimensions (_HYRect r, _HYRect rel){	_HYPullDown* theParent = (_HYPullDown*) this;	theParent->_HYPlatformComponent::_SetDimensions (r,rel);	_SetVisibleSize (rel);}//__________________________________________________________________void		_HYPlatformPullDown::_SetVisibleSize (_HYRect rel){	_HYPullDown* theParent = (_HYPullDown*) this;	menuRect.left = rel.left+3;	menuRect.bottom = rel.bottom;	menuRect.right= rel.right-3;	menuRect.top = rel.top;	if (myMenu)	{		if (menuRect.bottom-menuRect.top>25)			menuRect.bottom = menuRect.top+25;		if (menuRect.right-menuRect.left>menuWidth)			menuRect.right = menuRect.left+menuWidth;	}	AlignRectangle (rel, menuRect, theParent->GetAlignFlags());	SetWindowPos (myMenu,NULL,menuRect.left,menuRect.top,menuRect.right-menuRect.left+1,(menuRect.bottom-menuRect.top),SWP_NOZORDER);	SendMessage (myMenu,CB_SETCURSEL,selection,0L);		//if (theParent->IsEnabled())		ShowWindow(myMenu,SW_SHOW);}//__________________________________________________________________void		_HYPlatformPullDown::_EnableItem (long index, bool toggle){		EnableMenuItem (theMenu, index, MF_BYPOSITION|(toggle?MF_ENABLED:(MF_GRAYED|MF_DISABLED)));			}//__________________________________________________________________void		_HYPlatformPullDown::_Paint (Ptr p){	_HYPullDown * theParent = (_HYPullDown*)this;	_HYRect * relRect = (_HYRect*)p;	_RefreshComboBox();	if (!(theParent->settings.width&HY_COMPONENT_TRANSP_BG))	{		RECT    cRect;		cRect.left = relRect->left;		cRect.right = relRect->right;		cRect.top = relRect->top;		cRect.bottom = relRect->bottom;		HDC 	theContext = (HDC)relRect->width;		FillRect (theContext,&cRect,backFill);	}	UpdateWindow (myMenu);	theParent->_HYPlatformComponent::_Paint(p);}//__________________________________________________________________void		_HYPlatformPullDown::_EnableMenu 	 (bool flag){	if (myMenu)		EnableWindow (myMenu, flag);}//__________________________________________________________________ _HYRect	_HYPullDown::_SuggestDimensions (void) { 	_HYRect res = {25,100,25,100,HY_COMPONENT_NO_SCROLL}; 	if (myMenu) 		res.right = menuWidth; 	return res; } //__________________________________________________________________void	_HYPullDown::_SetMenuItemTextStyle (long ID, char style){ 	//if (myMenu) 		//SetItemStyle (myMenu,ID+1,style); 		    // TBI}//__________________________________________________________________bool _HYPullDown::_ProcessOSEvent (Ptr vEvent){	/*_HYWindowsUIMessage * theEvent = (_HYWindowsUIMessage *)vEvent;	switch (theEvent->iMsg)	{		case WM_CTLCOLORLISTBOX:		{			if ((HWND)theEvent->lParam == myMenu)			{				SetTextColor ((HDC)theEvent->wParam,GetSysColor (COLOR_GRAYTEXT));				return true;			}			break;		}		}*/	return _HYPlatformComponent::_ProcessOSEvent (vEvent);		}//EOF