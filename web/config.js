// Default values. Can be overridden at runtime via URL param `?api_base=` or saved in localStorage.
window.API_BASE = "https://your-backend.example.com"; // no trailing slash
window.API_KEY = "3JG1fMaUXwca8QC5X5uisQcfhmc_7idKNnfBMNyTWs5egSfSG"; // optional: set same value as server PUBLIC_API_KEY

(function(){
	try{
		const params = new URLSearchParams(window.location.search);
		const urlParam = params.get('api_base');
		const saved = localStorage.getItem('API_BASE');
		if(urlParam){
			window.API_BASE = urlParam;
			localStorage.setItem('API_BASE', urlParam);
		} else if(saved){
			window.API_BASE = saved;
		}
	}catch(e){
		// ignore
	}

	// Helper to set API_BASE from UI/console and persist to localStorage
	window.setApiBase = function(url){
		if(!url) return;
		window.API_BASE = url;
		try{ localStorage.setItem('API_BASE', url); }catch(e){}
		return window.API_BASE;
	};
	window.clearApiBase = function(){
		try{ localStorage.removeItem('API_BASE'); }catch(e){}
		return (window.API_BASE = null);
	};
})();
