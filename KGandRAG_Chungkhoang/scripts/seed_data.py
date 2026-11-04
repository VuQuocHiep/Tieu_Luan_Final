"""Tạo bộ dữ liệu học thuật về chứng khoán và vàng. Không chứa giá thị trường thời gian thực."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
SRC = {
 'SEC_STOCK': ('SEC Investor.gov', 'https://www.investor.gov/introduction-investing/investing-basics/investment-products/stocks'),
 'SEC_ALLOC': ('SEC Investor.gov', 'https://www.investor.gov/introduction-investing/getting-started/asset-allocation'),
 'SEC_PRODUCTS': ('SEC Investor.gov', 'https://www.investor.gov/introduction-investing/investing-basics/investment-products'),
 'SEC_ORDERS': ('SEC Investor.gov', 'https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-bulletins-15'),
 'WGC_INVEST': ('World Gold Council', 'https://www.gold.org/goldhub/how-to-invest'),
 'WGC_DRIVERS': ('World Gold Council', 'https://www.gold.org/goldhub/research/what-drives-gold'),
 'WGC_2026': ('World Gold Council', 'https://www.gold.org/goldhub/research/gold-mid-year-outlook-2026'),
}
# id, title, category, original Vietnamese explanation (not source verbatim), reference, linked entity ids
RAW = [
 ('D01','Cổ phiếu là gì?','Chứng khoán','Cổ phiếu là chứng khoán thể hiện quyền sở hữu một phần doanh nghiệp. Người nắm giữ có thể hưởng lợi từ tăng giá hoặc cổ tức nhưng cũng có nguy cơ thua lỗ khi giá giảm.','SEC_STOCK','stock equity company'),
 ('D02','Cổ phiếu phổ thông và ưu đãi','Chứng khoán','Cổ phiếu phổ thông thường đi kèm quyền biểu quyết, trong khi một số loại cổ phiếu ưu đãi có quyền ưu tiên về cổ tức hoặc thanh toán theo điều kiện phát hành. Quyền cụ thể phụ thuộc điều lệ và quy định.','SEC_STOCK','stock common preferred'),
 ('D03','Lợi nhuận cổ phiếu và cổ tức','Chứng khoán','Hai nguồn lợi nhuận quen thuộc khi nắm giữ cổ phiếu là phần tăng giá vốn và cổ tức được doanh nghiệp phân phối. Cổ tức không được bảo đảm; lợi nhuận tổng còn bị ảnh hưởng bởi thuế và phí.','SEC_STOCK','stock dividend return company'),
 ('D04','Rủi ro của đầu tư cổ phiếu','Rủi ro','Giá cổ phiếu có thể biến động do kết quả kinh doanh, lãi suất, kỳ vọng thị trường và sự kiện bất ngờ. Nhà đầu tư có thể mất một phần hoặc toàn bộ vốn; không nên xem mức sinh lời quá khứ là cam kết tương lai.','SEC_STOCK','stock volatility risk'),
 ('D05','Thanh khoản và khối lượng giao dịch','Thị trường','Thanh khoản phản ánh mức độ có thể mua bán tài sản mà không làm thay đổi giá quá nhiều. Cổ phiếu có khối lượng giao dịch thấp có thể đi kèm chênh lệch mua bán cao và khó khớp lệnh khi thị trường biến động.','SEC_PRODUCTS','stock liquidity spread'),
 ('D06','Chỉ số thị trường chứng khoán','Thị trường','Chỉ số chứng khoán theo dõi biến động giá của một rổ cổ phiếu theo phương pháp tính xác định. VN-Index là ví dụ về chỉ số đại diện cổ phiếu niêm yết trên HOSE; một chỉ số không phải mã cổ phiếu riêng lẻ.','SEC_PRODUCTS','index stock market'),
 ('D07','Quỹ ETF cổ phiếu','Chứng khoán','ETF là quỹ có chứng chỉ được giao dịch trên sàn; nhiều quỹ theo dõi chỉ số hoặc một rổ tài sản. ETF cổ phiếu có thể giúp tiếp cận nhiều doanh nghiệp nhưng vẫn chịu rủi ro thị trường và phí quản lý.','SEC_ALLOC','etf stock diversification'),
 ('D08','Đa dạng hóa danh mục','Quản trị danh mục','Đa dạng hóa là phân bổ vốn cho nhiều khoản đầu tư hoặc loại tài sản để tránh quá phụ thuộc vào một vị thế. Đa dạng hóa có thể giảm rủi ro tập trung nhưng không loại bỏ mọi thua lỗ.','SEC_ALLOC','diversification portfolio risk'),
 ('D09','Phân bổ tài sản','Quản trị danh mục','Phân bổ tài sản là lựa chọn tỷ trọng giữa cổ phiếu, trái phiếu, tiền mặt hoặc vàng tùy mục tiêu, thời gian nắm giữ và khả năng chịu rủi ro. Không có tỷ trọng tối ưu chung cho tất cả mọi người.','SEC_ALLOC','allocation stock gold portfolio'),
 ('D10','Khẩu vị rủi ro và thời gian đầu tư','Quản trị danh mục','Khả năng chấp nhận thua lỗ, thời hạn cần dùng tiền và mục tiêu tài chính ảnh hưởng đến cơ cấu danh mục. Vốn cần sử dụng trong ngắn hạn thường không thích hợp với vị thế biến động quá cao.','SEC_ALLOC','risk horizon allocation'),
 ('D11','Lệnh thị trường và lệnh giới hạn','Thị trường','Lệnh thị trường ưu tiên thực hiện giao dịch nhưng giá khớp có thể khác giá đang nhìn thấy. Lệnh giới hạn chỉ mua hoặc bán trong giới hạn giá đã đặt, đổi lại có thể không được khớp. Tên lệnh và quy tắc phụ thuộc sàn giao dịch.','SEC_ORDERS','order_market order_limit liquidity'),
 ('D12','Lệnh dừng và giới hạn thua lỗ','Thị trường','Lệnh dừng có thể kích hoạt thành lệnh thị trường khi chạm mức dừng; vì thế giá khớp thực tế không luôn bằng giá kích hoạt. Lệnh dừng giới hạn kiểm soát giá nhưng có nguy cơ không khớp khi thị trường biến động nhanh.','SEC_ORDERS','stop_order volatility risk'),
 ('D13','P/E và EPS','Phân tích','EPS là lợi nhuận tính trên một cổ phần theo định nghĩa kế toán tương ứng. P/E thường được tính bằng giá cổ phiếu chia EPS; hệ số này cần đặt trong bối cảnh ngành, tăng trưởng, chu kỳ và chất lượng lợi nhuận. EPS âm khiến P/E thông thường không có ý nghĩa đơn giản.','SEC_PRODUCTS','pe eps stock'),
 ('D14','Vốn hóa thị trường','Phân tích','Vốn hóa thị trường của công ty niêm yết được ước tính bằng giá một cổ phiếu nhân số cổ phiếu đang lưu hành tương ứng. Giá một cổ phiếu cao không tự chứng minh doanh nghiệp lớn hơn doanh nghiệp có giá một cổ phiếu thấp.','SEC_STOCK','marketcap stock company'),
 ('D15','Phân tích cơ bản và kỹ thuật','Phân tích','Phân tích cơ bản nghiên cứu doanh nghiệp, ngành và báo cáo tài chính; phân tích kỹ thuật khảo sát dữ liệu giá và khối lượng. Cả hai là công cụ hỗ trợ đánh giá, không bảo đảm dự báo chính xác chiều biến động.','SEC_PRODUCTS','fundamental technical stock'),
 ('D16','Lãi suất ảnh hưởng cổ phiếu','Vĩ mô','Lãi suất có thể tác động đến chi phí vay, lãi suất chiết khấu và quyết định tiêu dùng. Khi lợi suất tăng, định giá cổ phiếu có thể chịu áp lực, nhưng tác động thực tế phụ thuộc tăng trưởng, lợi nhuận doanh nghiệp và kỳ vọng thị trường.','SEC_ALLOC','interest stock valuation'),
 ('D17','Báo cáo tài chính và lợi nhuận','Phân tích','Đọc doanh thu, lợi nhuận, dòng tiền, nợ và thuyết minh giúp hiểu tình hình doanh nghiệp. Tăng trưởng doanh thu không đồng nghĩa lợi nhuận hay dòng tiền sẽ tăng tương ứng.','SEC_STOCK','financials company fundamental'),
 ('D18','Chi phí giao dịch và chênh lệch giá','Thị trường','Chi phí đầu tư có thể bao gồm phí giao dịch, phí quản lý, thuế theo quy định và chênh lệch giá mua - bán. Cần tính cả chi phí khi so sánh lợi nhuận ròng giữa các hình thức đầu tư.','SEC_PRODUCTS','fees spread return'),
 ('D19','IPO và thị trường thứ cấp','Chứng khoán','IPO là lần chào bán cổ phiếu lần đầu ra công chúng theo cơ chế áp dụng. Sau phát hành, cổ phiếu có thể được giao dịch trên thị trường thứ cấp nếu đáp ứng điều kiện; IPO không đồng nghĩa giá sẽ tăng.','SEC_STOCK','ipo stock market'),
 ('D20','Vàng là tài sản đầu tư gì?','Vàng','Vàng là kim loại quý được giao dịch dưới dạng vật chất hoặc qua sản phẩm tài chính. Vàng thường được xem như tài sản lưu trữ giá trị và đa dạng hóa danh mục, song giá vẫn có thể dao động mạnh.','WGC_INVEST','gold asset risk'),
 ('D21','Giá vàng giao ngay XAU/USD','Vàng','XAU/USD biểu diễn giá vàng quốc tế so với USD, thường niêm yết theo một troy ounce. Giá vàng quốc tế không thể quy đổi trực tiếp thành giá bán lẻ vàng trong nước nếu bỏ qua tỷ giá, đơn vị, chênh lệch và chi phí.','WGC_INVEST','gold xau usd exchange'),
 ('D22','Vàng trong nước và giá thế giới','Vàng','Giá vàng vật chất trong nước có thể khác giá quốc tế do cung cầu địa phương, tỷ giá, hình thức sản phẩm, chi phí lưu thông và điều kiện thị trường. So sánh phải chuẩn hóa khối lượng, độ tinh khiết và thời điểm báo giá.','WGC_INVEST','gold domestic exchange spread'),
 ('D23','Chênh lệch giá mua và bán vàng','Vàng','Giá mua vào là mức đơn vị kinh doanh trả khi mua lại vàng; giá bán ra là mức họ bán cho khách. Chênh lệch giá mua - bán là chi phí ngầm quan trọng: mua rồi bán ngay thường lỗ chênh lệch dù giá tham chiếu chưa đổi.','WGC_INVEST','gold spread fees'),
 ('D24','Vàng miếng, vàng nhẫn và lưu trữ','Vàng','Mua vàng vật chất cần chú ý loại sản phẩm, độ tinh khiết, chứng từ, uy tín người bán, lưu trữ và bảo hiểm khi thích hợp. Vàng vật chất có rủi ro thất lạc, mất cắp và chi phí bảo quản.','WGC_INVEST','gold bullion storage'),
 ('D25','Quỹ ETF vàng','Vàng','ETF vàng được bảo chứng bằng vàng vật chất là công cụ tài chính giao dịch trên sàn, thường nhằm theo dõi biến động giá vàng sau chi phí. Nắm chứng chỉ quỹ không tương đương trực tiếp cầm giữ vàng vật chất; cần xem điều khoản và khả năng tiếp cận tại thị trường của bạn.','WGC_INVEST','gold etfgold fees'),
 ('D26','Hợp đồng tương lai vàng','Vàng','Hợp đồng tương lai vàng quy định giao dịch một lượng vàng theo điều kiện xác định ở thời điểm tương lai. Đòn bẩy và ký quỹ có thể khuếch đại cả lợi nhuận lẫn thua lỗ; sản phẩm không phù hợp với mọi nhà đầu tư.','WGC_INVEST','gold futures leverage risk'),
 ('D27','USD và giá vàng','Vĩ mô','Giá vàng quốc tế thường được niêm yết bằng USD. Đồng USD và tỷ giá có thể ảnh hưởng sức mua của người nắm các đồng tiền khác; mối quan hệ USD - vàng không cố định từng ngày và chịu nhiều yếu tố đồng thời.','WGC_DRIVERS','gold usd exchange'),
 ('D28','Lãi suất thực và vàng','Vĩ mô','Vàng vật chất không tự chi trả lãi suất định kỳ. Khi lợi suất thực của tài sản sinh lãi tăng, chi phí cơ hội nắm giữ vàng có thể tăng; ngược lại lợi suất thực giảm có thể hỗ trợ nhu cầu vàng. Đây là xu hướng thường gặp, không phải quy luật bất biến.','WGC_DRIVERS','gold realyield interest'),
 ('D29','Lạm phát và vàng','Vĩ mô','Vàng đôi khi được dùng để phân tán rủi ro mất sức mua dài hạn, nhưng vàng không luôn tăng ngay khi chỉ số giá tiêu dùng tăng. Trong ngắn hạn, USD, lãi suất thực và dòng vốn có thể lấn át tác động lạm phát.','WGC_DRIVERS','gold inflation realyield'),
 ('D30','Rủi ro địa chính trị và vàng','Vĩ mô','Bất ổn địa chính trị hoặc kinh tế có thể thúc đẩy nhu cầu phòng vệ bằng vàng. Tuy vậy, phản ứng giá phụ thuộc kỳ vọng trước đó, USD, lãi suất và hoạt động chốt lời, nên không bảo đảm mọi khủng hoảng đều làm giá vàng tăng.','WGC_2026','gold geopolitical safehaven'),
 ('D31','Ngân hàng trung ương mua vàng','Vàng','Ngân hàng trung ương có thể nắm giữ vàng trong dự trữ quốc tế. Hoạt động mua hoặc bán của khu vực chính thức là một trong các yếu tố ảnh hưởng cung cầu vàng toàn cầu, không phải tín hiệu dự báo chắc chắn giá ngắn hạn.','WGC_2026','gold centralbank demand'),
 ('D32','Vàng có trả cổ tức không?','Vàng','Vàng vật chất không tạo dòng tiền nội tại như cổ tức từ cổ phiếu hay coupon trái phiếu. Kết quả đầu tư phụ thuộc chủ yếu biến động giá sau khi trừ chênh lệch và chi phí lưu trữ; một số sản phẩm vàng tài chính có cấu trúc phí riêng.','WGC_INVEST','gold dividend fees'),
 ('D33','So sánh vàng và cổ phiếu','So sánh','Cổ phiếu đại diện quyền sở hữu doanh nghiệp và có thể tạo thu nhập cổ tức; vàng là tài sản hữu hình không tạo dòng tiền từ hoạt động kinh doanh. Hai nhóm có động lực giá và rủi ro khác nhau, vì vậy không thể khẳng định một loại luôn tốt hơn loại kia.','SEC_ALLOC','gold stock risk dividend'),
 ('D34','Kết hợp vàng và cổ phiếu','So sánh','Kết hợp vàng, cổ phiếu và các tài sản khác có thể làm phân tán nguồn rủi ro trong danh mục. Tác dụng đa dạng hóa phụ thuộc tỷ trọng, tương quan biến động và giai đoạn thị trường; không có công thức tỷ trọng chung phù hợp với mọi nhà đầu tư.','SEC_ALLOC','portfolio gold stock diversification'),
 ('D35','Rủi ro tỷ giá khi giữ vàng','Rủi ro','Người đầu tư bằng VND nhưng theo dõi vàng định giá USD phải phân biệt biến động giá vàng quốc tế và biến động tỷ giá USD/VND. Ngoài ra còn có chênh lệch thị trường địa phương và chi phí mua bán.','WGC_DRIVERS','gold exchange usd domestic'),
 ('D36','Tương quan cổ phiếu và vàng','So sánh','Tương quan là thước đo sự dịch chuyển cùng chiều hoặc ngược chiều giữa hai chuỗi lợi suất, thường từ -1 tới +1. Tương quan vàng và cổ phiếu thay đổi theo thời gian nên không nên giả định luôn âm hoặc luôn phòng vệ hiệu quả.','WGC_INVEST','gold stock correlation diversification'),
 ('D37','Quản lý biến động và kỷ luật','Quản trị danh mục','Xác định mục tiêu, quỹ dự phòng, quy mô vị thế và nguyên tắc tái cân bằng có thể giúp tránh quyết định bốc đồng khi giá biến động. Biến động cao có thể tạo cả cơ hội lẫn nguy cơ lỗ lớn.','SEC_ALLOC','volatility portfolio risk'),
 ('D38','Đầu tư định kỳ (DCA)','Quản trị danh mục','Đầu tư định kỳ là phân bổ khoản vốn bằng nhau theo lịch thay vì cố dự đoán một điểm mua tối ưu. Phương pháp có thể giảm áp lực chọn thời điểm nhưng không bảo vệ trước xu hướng giá giảm dài hạn hoặc sản phẩm kém chất lượng.','SEC_ALLOC','dca portfolio horizon'),
 ('D39','Cảnh giác lừa đảo đầu tư','Rủi ro','Lời hứa lợi nhuận cao không rủi ro, thúc giục chuyển tiền gấp, trang web giả mạo và yêu cầu cung cấp mã OTP là dấu hiệu cần cảnh giác. Luôn kiểm tra tư cách đơn vị cung cấp và nguồn công bố trước khi giao dịch.','SEC_PRODUCTS','fraud risk fees'),
 ('D40','Giá hôm nay và dữ liệu thời gian thực','Dữ liệu','Bộ Knowledge Graph + RAG này chỉ có tài liệu kiến thức tĩnh, không kết nối bảng giá hoặc API thị trường. Mọi câu hỏi về giá vàng hôm nay, giá mã chứng khoán hiện tại, xu hướng ngày mai hoặc khuyến nghị mua bán ngay cần nguồn dữ liệu cập nhật có thời gian và đơn vị rõ ràng.','SEC_PRODUCTS','gold stock realtime'),
]
NODES = [
 ('stock','Cổ phiếu','Chứng khoán','chứng khoán|cổ phiếu|shares|stock'),('equity','Quyền sở hữu','Khái niệm','quyền sở hữu|vốn chủ sở hữu'),('company','Doanh nghiệp','Chủ thể','công ty|doanh nghiệp'),('common','Cổ phiếu phổ thông','Chứng khoán','phổ thông|common stock'),('preferred','Cổ phiếu ưu đãi','Chứng khoán','ưu đãi|preferred stock'),('dividend','Cổ tức','Chứng khoán','cổ tức|dividend'),('return','Lợi suất','Chỉ số','lợi suất|lợi nhuận|sinh lời'),('risk','Rủi ro đầu tư','Rủi ro','rủi ro|thua lỗ'),('volatility','Biến động giá','Rủi ro','biến động|dao động'),('liquidity','Thanh khoản','Chỉ số','thanh khoản|khối lượng giao dịch'),('spread','Chênh lệch mua bán','Chi phí','spread|chênh lệch|giá mua bán'),('index','Chỉ số chứng khoán','Thị trường','chỉ số|vn-index|vn index'),('market','Thị trường','Thị trường','sàn chứng khoán|thị trường'),('etf','ETF cổ phiếu','Chứng khoán','etf cổ phiếu|quỹ chỉ số|quỹ etf'),('diversification','Đa dạng hóa','Quản trị','đa dạng hóa|phân tán rủi ro'),('portfolio','Danh mục đầu tư','Quản trị','danh mục|portfolio|tái cân bằng|tái cân bằng vốn'),('allocation','Phân bổ tài sản','Quản trị','phân bổ vốn|phân bổ tài sản'),('horizon','Thời hạn đầu tư','Quản trị','dài hạn|ngắn hạn|thời hạn'),('order_market','Lệnh thị trường','Giao dịch','lệnh thị trường|market order|lệnh mp'),('order_limit','Lệnh giới hạn','Giao dịch','lệnh giới hạn|limit order|lệnh lo'),('stop_order','Lệnh dừng','Giao dịch','stop loss|stop-limit|lệnh dừng|cắt lỗ'),('pe','P/E','Phân tích','p/e|pe ratio|hệ số pe'),('eps','EPS','Phân tích','eps|lãi trên cổ phiếu'),('marketcap','Vốn hóa','Phân tích','vốn hóa|market cap'),('fundamental','Phân tích cơ bản','Phân tích','phân tích cơ bản|fundamental'),('technical','Phân tích kỹ thuật','Phân tích','phân tích kỹ thuật|technical'),('valuation','Định giá','Phân tích','định giá|valuation'),('interest','Lãi suất','Vĩ mô','lãi suất|interest rate'),('financials','Báo cáo tài chính','Phân tích','báo cáo tài chính|doanh thu|dòng tiền'),('fees','Chi phí đầu tư','Chi phí','phí giao dịch|phí quản lý|chi phí'),('ipo','IPO','Chứng khoán','ipo|phát hành lần đầu'),('gold','Vàng','Vàng','vàng|kim loại quý|gold'),('asset','Tài sản đầu tư','Khái niệm','tài sản|asset'),('xau','XAU/USD','Vàng','xau/usd|xauusd|giá vàng thế giới|spot gold'),('usd','USD','Vĩ mô','usd|đô la mỹ|đồng đô'),('exchange','Tỷ giá','Vĩ mô','tỷ giá|usd/vnd|ngoại tệ'),('domestic','Giá vàng trong nước','Vàng','giá vàng trong nước|vàng sjc|vàng nội địa'),('bullion','Vàng vật chất','Vàng','vàng miếng|vàng nhẫn|vàng vật chất|bullion'),('storage','Lưu trữ vàng','Rủi ro','cất giữ|bảo quản|lưu trữ vàng'),('etfgold','ETF vàng','Vàng','etf vàng|quỹ vàng|gold etf'),('futures','Hợp đồng tương lai vàng','Vàng','futures vàng|phái sinh vàng|hợp đồng tương lai vàng'),('leverage','Đòn bẩy','Rủi ro','đòn bẩy|ký quỹ|margin'),('realyield','Lãi suất thực','Vĩ mô','lãi suất thực|lợi suất thực'),('inflation','Lạm phát','Vĩ mô','lạm phát|cpi'),('geopolitical','Địa chính trị','Vĩ mô','địa chính trị|chiến tranh|bất ổn'),('safehaven','Tài sản trú ẩn','Vàng','trú ẩn|safe haven'),('centralbank','Ngân hàng trung ương','Chủ thể','ngân hàng trung ương|dự trữ vàng'),('demand','Cung cầu vàng','Vàng','cung cầu vàng|nhu cầu vàng'),('correlation','Tương quan','Chỉ số','tương quan|correlation'),('dca','Đầu tư định kỳ','Quản trị','dca|trung bình giá|đầu tư định kỳ'),('fraud','Lừa đảo đầu tư','Rủi ro','lừa đảo|otp|giả mạo|lợi nhuận cam kết'),('realtime','Dữ liệu thời gian thực','Dữ liệu','thời gian thực|hôm nay|giá hiện tại|giá trực tiếp'),
]
LINKS = [
 ('stock','equity','thể hiện'),('stock','company','đại diện một phần'),('stock','common','có loại'),('stock','preferred','có loại'),('stock','dividend','có thể trả'),('stock','return','có thể tạo'),('stock','risk','có'),('stock','volatility','chịu ảnh hưởng'),('stock','liquidity','có đặc tính'),('stock','index','được tổng hợp trong'),('stock','market','giao dịch trên'),('stock','etf','có thể nằm trong'),('etf','diversification','có thể hỗ trợ'),('diversification','portfolio','áp dụng cho'),('allocation','portfolio','cấu thành'),('allocation','horizon','phụ thuộc'),('risk','horizon','đánh giá cùng'),('order_market','market','được sử dụng trên'),('order_market','liquidity','phụ thuộc'),('order_limit','market','được sử dụng trên'),('stop_order','volatility','bị ảnh hưởng'),('stock','pe','có chỉ số định giá'),('pe','eps','dùng mẫu số'),('pe','valuation','hỗ trợ'),('stock','marketcap','có chỉ số'),('company','financials','công bố'),('fundamental','financials','nghiên cứu'),('technical','volatility','nghiên cứu'),('valuation','interest','chịu tác động'),('company','ipo','có thể thực hiện'),('stock','fees','phát sinh'),('order_limit','spread','cân nhắc'),('gold','asset','là loại'),('gold','xau','có giá tham chiếu'),('gold','bullion','giao dịch dưới dạng'),('gold','etfgold','có sản phẩm'),('gold','futures','có sản phẩm'),('gold','risk','có'),('gold','domestic','có báo giá'),('gold','spread','phát sinh'),('bullion','storage','cần'),('etfgold','fees','chịu'),('futures','leverage','có thể sử dụng'),('gold','usd','thường định giá bằng'),('xau','usd','được niêm yết bằng'),('domestic','exchange','chịu ảnh hưởng'),('domestic','spread','có chênh lệch'),('usd','exchange','liên quan'),('gold','realyield','chịu ảnh hưởng'),('realyield','interest','liên quan'),('gold','inflation','chịu ảnh hưởng'),('inflation','realyield','liên quan'),('gold','geopolitical','chịu ảnh hưởng'),('gold','safehaven','thường được xem là'),('gold','centralbank','được nắm giữ bởi'),('centralbank','demand','tạo một phần'),('gold','demand','chịu ảnh hưởng'),('gold','dividend','không tự trả'),('stock','gold','có thể phân bổ cùng'),('gold','portfolio','có thể thuộc'),('gold','diversification','có thể hỗ trợ'),('gold','correlation','có tương quan với cổ phiếu'),('correlation','stock','được tính với'),('correlation','diversification','ảnh hưởng'),('portfolio','dca','có thể áp dụng'),('risk','fraud','bao gồm'),('gold','realtime','cần tra giá qua'),('stock','realtime','cần tra giá qua'),('return','fees','cần khấu trừ'),('interest','stock','có thể ảnh hưởng'),('interest','gold','có thể ảnh hưởng'),('diversification','risk','có thể hạn chế tập trung'),('volatility','risk','là thành phần'),('leverage','risk','khuếch đại'),
]

def main():
 docs = [{'id':id,'title':title,'category':category,'text':content,'source_name':SRC[source][0], 'source_url':SRC[source][1], 'entities':entities.split()} for id,title,category,content,source,entities in RAW]
 nodes = [{'id':id,'label':label,'type':kind,'aliases':aliases.split('|')} for id,label,kind,aliases in NODES]
 ids={n['id'] for n in nodes}
 assert all(set(d['entities'])<=ids for d in docs)
 edges=[]
 for number,(src,dst,relation) in enumerate(LINKS,1):
  assert src in ids and dst in ids
  supports=[d['id'] for d in docs if src in d['entities'] and dst in d['entities']]
  if not supports: supports=[d['id'] for d in docs if src in d['entities']][:2]
  if not supports: supports=[d['id'] for d in docs if dst in d['entities']][:2]
  edges.append({'id':f'E{number:03d}','source':src,'target':dst,'relation':relation,'doc_ids':supports})
 (ROOT/'data').mkdir(exist_ok=True)
 for filename, payload in [('documents.json',docs),('graph.json',{'nodes':nodes,'edges':edges})]:
  (ROOT/'data'/filename).write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 # Q labels are defined by the author based on the same corpus: in-domain, not independent validation.
 questions=[
 'Cổ phiếu là gì, quyền sở hữu doanh nghiệp ra sao?',
 'Cổ phiếu phổ thông khác cổ phiếu ưu đãi thế nào?',
 'Cổ tức và phần tăng giá cổ phiếu tạo lợi nhuận ra sao?',
 'Đầu tư cổ phiếu có thể chịu những rủi ro gì?',
 'Thanh khoản và khối lượng giao dịch là gì?',
 'VN-Index là chỉ số chứng khoán gì?',
 'ETF cổ phiếu là gì và hoạt động ra sao?',
 'Vì sao cần đa dạng hóa danh mục đầu tư?',
 'Phân bổ tài sản giữa cổ phiếu vàng và tiền mặt thế nào?',
 'Khẩu vị rủi ro và thời hạn đầu tư quan trọng ra sao?',
 'Lệnh thị trường khác lệnh giới hạn như thế nào?',
 'Lệnh dừng stop loss hoạt động ra sao?',
 'Hệ số P/E và EPS có ý nghĩa gì?',
 'Vốn hóa thị trường của doanh nghiệp được tính thế nào?',
 'Phân tích cơ bản khác phân tích kỹ thuật điểm nào?',
 'Lãi suất tăng có thể tác động đến giá cổ phiếu như thế nào?',
 'Đọc báo cáo tài chính và dòng tiền để phân tích doanh nghiệp?',
 'Phí giao dịch và chênh lệch mua bán ảnh hưởng lợi nhuận thế nào?',
 'IPO là gì và liên quan đến thị trường thứ cấp ra sao?',
 'Vàng có đặc điểm nào khi là tài sản đầu tư?',
 'XAU/USD là gì, đơn vị giá vàng quốc tế?',
 'Vì sao giá vàng trong nước khác giá vàng thế giới?',
 'Chênh lệch giá vàng mua vào bán ra nghĩa là gì?',
 'Cần lưu ý gì khi mua vàng miếng và vàng nhẫn vật chất?',
 'ETF vàng là gì, khác cầm vàng vật chất thế nào?',
 'Hợp đồng tương lai vàng có rủi ro đòn bẩy gì?',
 'USD mạnh lên tác động đến giá vàng ra sao?',
 'Lãi suất thực ảnh hưởng giá vàng bằng cơ chế nào?',
 'Vàng có luôn tăng theo lạm phát không?',
 'Bất ổn địa chính trị ảnh hưởng nhu cầu trú ẩn vàng ra sao?',
 'Ngân hàng trung ương mua dự trữ vàng làm gì?',
 'Giữ vàng có được nhận cổ tức không?',
 'Điểm khác nhau khi đầu tư vàng và cổ phiếu?',
 'Kết hợp vàng cổ phiếu giúp đa dạng hóa danh mục ra sao?',
 'Rủi ro tỷ giá USD/VND khi mua vàng tính thế nào?',
 'Tương quan lợi suất giữa cổ phiếu và vàng có cố định không?',
 'Nên quản lý biến động thị trường và tái cân bằng danh mục ra sao?',
 'Đầu tư định kỳ DCA có lợi ích và hạn chế gì?',
 'Cần nhận biết dấu hiệu lừa đảo đầu tư nào?',
 'Hệ thống có biết giá vàng hôm nay và giá cổ phiếu hiện tại không?',
 ]
 assert len(questions)==len(docs)
 data=[{'id':f'Q{i:02d}','question':q,'gold_docs':[f'D{i:02d}']} for i,q in enumerate(questions,1)]
 # One validated reference chunk per question makes manual inspection straightforward.
 (ROOT/'eval'/'questions.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 print(f'Generated: {len(docs)} knowledge passages, {len(nodes)} graph nodes, {len(edges)} edges, {len(data)} evaluation questions.')

if __name__ == '__main__': main()
