module class9_tree21(input wire [467:0] i, output wire [0:0] o);

reg [0:0] o_reg;
assign o = o_reg;

always @(*) begin
	if(((~i[321]) & (~i[200])) | ((~i[280]) & i[347] & i[200]) | (i[108] & (~i[347]) & i[200]) | ((~i[108]) & (~i[347]) & i[200]) | (i[454] & i[394] & i[280] & i[347] & i[200]) | ((~i[454]) & i[394] & i[280] & i[347] & i[200]) | (i[185] & (~i[394]) & i[280] & i[347] & i[200]) | ((~i[185]) & (~i[394]) & i[280] & i[347] & i[200]))
		o_reg = 0;
	else
		o_reg = 1;
end

endmodule
