module class6_tree10(input wire [467:0] i, output wire [0:0] o);

reg [0:0] o_reg;
assign o = o_reg;

always @(*) begin
	if(((~i[154]) & i[151] & i[170]) | (i[185] & (~i[83]) & (~i[170])) | (i[361] & i[359] & i[83] & (~i[170])) | (i[120] & (~i[406]) & i[154] & i[151] & i[170]) | (i[76] & i[177] & i[438] & (~i[151]) & i[170]) | (i[244] & i[246] & (~i[359]) & i[83] & (~i[170])) | (i[11] & i[30] & (~i[185]) & (~i[83]) & (~i[170])))
		o_reg = 0;
	else
		o_reg = 1;
end

endmodule
